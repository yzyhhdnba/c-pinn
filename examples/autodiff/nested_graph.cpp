#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

namespace mini_ad {

struct Node;
using NodePtr = std::shared_ptr<Node>;
using VjpFn = std::function<std::vector<NodePtr>(const NodePtr&, bool)>;

struct Node {
    int id = 0;
    double value = 0.0;
    bool requires_grad = false;
    bool is_leaf = false;
    double grad = 0.0;
    std::string op;
    std::vector<NodePtr> parents;
    VjpFn vjp;
};

int next_node_id() {
    static int id = 0;
    return ++id;
}

NodePtr make_node(double value,
                  bool requires_grad,
                  const std::string& op,
                  const std::vector<NodePtr>& parents = {},
                  VjpFn vjp = nullptr,
                  bool is_leaf = false) {
    auto node = std::make_shared<Node>();
    node->id = next_node_id();
    node->value = value;
    node->requires_grad = requires_grad;
    node->is_leaf = is_leaf;
    node->grad = 0.0;
    node->op = op;
    node->parents = parents;
    node->vjp = std::move(vjp);
    return node;
}

NodePtr constant(double value) {
    return make_node(value, false, "const");
}

NodePtr variable(double value, bool requires_grad = true) {
    return make_node(value, requires_grad, "leaf", {}, nullptr, true);
}

NodePtr add(const NodePtr& a, const NodePtr& b);
NodePtr mul(const NodePtr& a, const NodePtr& b);
NodePtr neg(const NodePtr& a);
NodePtr sub(const NodePtr& a, const NodePtr& b);
NodePtr sin(const NodePtr& a);
NodePtr cos(const NodePtr& a);
NodePtr tanh(const NodePtr& a);

NodePtr add(const NodePtr& a, const NodePtr& b) {
    const bool req = a->requires_grad || b->requires_grad;
    auto out = make_node(a->value + b->value, req, "add", {a, b});
    out->vjp = [](const NodePtr& upstream, bool create_graph) {
        if (create_graph) {
            return std::vector<NodePtr>{upstream, upstream};
        }
        return std::vector<NodePtr>{constant(upstream->value), constant(upstream->value)};
    };
    return out;
}

NodePtr mul(const NodePtr& a, const NodePtr& b) {
    const bool req = a->requires_grad || b->requires_grad;
    auto out = make_node(a->value * b->value, req, "mul", {a, b});
    out->vjp = [a, b](const NodePtr& upstream, bool create_graph) {
        if (create_graph) {
            return std::vector<NodePtr>{mul(upstream, b), mul(upstream, a)};
        }
        return std::vector<NodePtr>{
            constant(upstream->value * b->value),
            constant(upstream->value * a->value),
        };
    };
    return out;
}

NodePtr neg(const NodePtr& a) {
    return mul(constant(-1.0), a);
}

NodePtr sub(const NodePtr& a, const NodePtr& b) {
    return add(a, neg(b));
}

NodePtr sin(const NodePtr& a) {
    const bool req = a->requires_grad;
    auto out = make_node(std::sin(a->value), req, "sin", {a});
    out->vjp = [a](const NodePtr& upstream, bool create_graph) {
        if (create_graph) {
            return std::vector<NodePtr>{mul(upstream, cos(a))};
        }
        return std::vector<NodePtr>{constant(upstream->value * std::cos(a->value))};
    };
    return out;
}

NodePtr cos(const NodePtr& a) {
    const bool req = a->requires_grad;
    auto out = make_node(std::cos(a->value), req, "cos", {a});
    out->vjp = [a](const NodePtr& upstream, bool create_graph) {
        if (create_graph) {
            return std::vector<NodePtr>{mul(upstream, neg(sin(a)))};
        }
        return std::vector<NodePtr>{constant(upstream->value * (-std::sin(a->value)))};
    };
    return out;
}

NodePtr tanh(const NodePtr& a) {
    const bool req = a->requires_grad;
    const double out_val = std::tanh(a->value);
    auto out = make_node(out_val, req, "tanh", {a});
    out->vjp = [out, out_val](const NodePtr& upstream, bool create_graph) {
        if (create_graph) {
            auto one = constant(1.0);
            auto deriv = sub(one, mul(out, out));
            return std::vector<NodePtr>{mul(upstream, deriv)};
        }
        return std::vector<NodePtr>{constant(upstream->value * (1.0 - out_val * out_val))};
    };
    return out;
}

void build_topo(const NodePtr& node,
                std::unordered_set<int>& visited,
                std::vector<NodePtr>& topo) {
    if (!node || visited.find(node->id) != visited.end()) {
        return;
    }
    visited.insert(node->id);
    for (const auto& p : node->parents) {
        build_topo(p, visited, topo);
    }
    topo.push_back(node);
}

std::vector<NodePtr> topo_sort(const NodePtr& root) {
    std::unordered_set<int> visited;
    std::vector<NodePtr> topo;
    build_topo(root, visited, topo);
    return topo;
}

struct GraphStats {
    std::size_t nodes = 0;
    std::size_t edges = 0;
};

GraphStats graph_stats(const NodePtr& root) {
    auto topo = topo_sort(root);
    std::size_t edges = 0;
    for (const auto& n : topo) {
        edges += n->parents.size();
    }
    return GraphStats{topo.size(), edges};
}

NodePtr grad(const NodePtr& output, const NodePtr& input, bool create_graph) {
    auto topo = topo_sort(output);
    std::unordered_map<int, NodePtr> grad_map;
    grad_map[output->id] = constant(1.0);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const NodePtr& node = *it;
        auto g_it = grad_map.find(node->id);
        if (g_it == grad_map.end()) {
            continue;
        }
        if (!node->vjp) {
            continue;
        }

        const NodePtr& upstream = g_it->second;
        auto pgrads = node->vjp(upstream, create_graph);

        for (std::size_t i = 0; i < node->parents.size(); ++i) {
            const NodePtr& parent = node->parents[i];
            if (!parent->requires_grad) {
                continue;
            }

            NodePtr pg = pgrads[i];
            if (!create_graph) {
                pg = constant(pg->value);
            }

            auto old_it = grad_map.find(parent->id);
            if (old_it == grad_map.end()) {
                grad_map[parent->id] = pg;
            } else {
                if (create_graph) {
                    grad_map[parent->id] = add(old_it->second, pg);
                } else {
                    grad_map[parent->id] = constant(old_it->second->value + pg->value);
                }
            }
        }
    }

    auto out_it = grad_map.find(input->id);
    if (out_it == grad_map.end()) {
        return constant(0.0);
    }
    if (create_graph) {
        return out_it->second;
    }
    return constant(out_it->second->value);
}

std::unordered_map<int, NodePtr> backward_collect(const NodePtr& output, bool create_graph) {
    auto topo = topo_sort(output);
    std::unordered_map<int, NodePtr> grad_map;
    grad_map[output->id] = constant(1.0);

    for (auto it = topo.rbegin(); it != topo.rend(); ++it) {
        const NodePtr& node = *it;
        auto g_it = grad_map.find(node->id);
        if (g_it == grad_map.end() || !node->vjp) {
            continue;
        }

        const NodePtr& upstream = g_it->second;
        auto pgrads = node->vjp(upstream, create_graph);

        for (std::size_t i = 0; i < node->parents.size(); ++i) {
            const NodePtr& parent = node->parents[i];
            if (!parent->requires_grad) {
                continue;
            }

            NodePtr pg = pgrads[i];
            if (!create_graph) {
                pg = constant(pg->value);
            }

            auto old_it = grad_map.find(parent->id);
            if (old_it == grad_map.end()) {
                grad_map[parent->id] = pg;
            } else if (create_graph) {
                grad_map[parent->id] = add(old_it->second, pg);
            } else {
                grad_map[parent->id] = constant(old_it->second->value + pg->value);
            }
        }
    }

    return grad_map;
}

void zero_grad(const std::vector<NodePtr>& params) {
    for (const auto& p : params) {
        if (p && p->is_leaf && p->requires_grad) {
            p->grad = 0.0;
        }
    }
}

void backward(const NodePtr& output, const std::vector<NodePtr>& params) {
    auto grad_map = backward_collect(output, false);
    for (const auto& p : params) {
        if (!p || !p->is_leaf || !p->requires_grad) {
            continue;
        }
        auto it = grad_map.find(p->id);
        if (it != grad_map.end()) {
            p->grad += it->second->value;
        }
    }
}

void print_graph(const NodePtr& root, const std::string& name, std::size_t max_nodes = 12) {
    auto topo = topo_sort(root);
    std::size_t edges = 0;
    for (const auto& n : topo) {
        edges += n->parents.size();
    }

    std::cout << "\n[Graph] " << name
              << " | nodes=" << topo.size()
              << " edges=" << edges << '\n';

    std::size_t shown = std::min(max_nodes, topo.size());
    for (std::size_t i = 0; i < shown; ++i) {
        const auto& n = topo[topo.size() - 1 - i];
        std::cout << "  node#" << n->id
                  << " op=" << n->op
                  << " val=" << std::setprecision(10) << n->value
                  << " req=" << (n->requires_grad ? "T" : "F")
                  << " parents=[";
        for (std::size_t j = 0; j < n->parents.size(); ++j) {
            std::cout << n->parents[j]->id;
            if (j + 1 < n->parents.size()) {
                std::cout << ',';
            }
        }
        std::cout << "]\n";
    }
    if (shown < topo.size()) {
        std::cout << "  ... (" << (topo.size() - shown) << " more nodes)\n";
    }
}

}  // namespace mini_ad

int main(int argc, char** argv) {
    using namespace mini_ad;

    double x0 = 0.5;
    bool json_mode = false;
    bool print_graphs = true;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            json_mode = true;
            print_graphs = false;
            continue;
        }
        if (arg == "--no-graphs") {
            print_graphs = false;
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: example_autodiff_nested_graph [x] [--json] [--no-graphs]\n";
            return 0;
        }
        try {
            x0 = std::stod(arg);
        } catch (const std::exception&) {
            std::cerr << "Invalid argument: " << arg << "\n";
            std::cerr << "Use --help for usage.\n";
            return 2;
        }
    }

    auto x = variable(x0, true);

    auto x2 = mul(x, x);
    auto x3 = mul(x2, x);
    auto y = add(x3, sin(x));

    auto dy_dx_plain = grad(y, x, false);
    auto dy_dx_graph = grad(y, x, true);
    auto d2y_dx2 = grad(dy_dx_graph, x, false);

    const double expected_d1 = 3.0 * x0 * x0 + std::cos(x0);
    const double expected_d2 = 6.0 * x0 - std::sin(x0);
    const double abs_err_d1 = std::abs(dy_dx_plain->value - expected_d1);
    const double abs_err_d2 = std::abs(d2y_dx2->value - expected_d2);

    const auto g0 = graph_stats(y);
    const auto g1 = graph_stats(dy_dx_graph);

    if (json_mode) {
        std::cout << std::fixed << std::setprecision(15);
        std::cout << "{";
        std::cout << "\"x\":" << x0 << ",";
        std::cout << "\"y\":" << y->value << ",";
        std::cout << "\"dy_dx_detached\":" << dy_dx_plain->value << ",";
        std::cout << "\"dy_dx_graph\":" << dy_dx_graph->value << ",";
        std::cout << "\"d2y_dx2\":" << d2y_dx2->value << ",";
        std::cout << "\"expected_d1\":" << expected_d1 << ",";
        std::cout << "\"expected_d2\":" << expected_d2 << ",";
        std::cout << "\"abs_err_d1\":" << abs_err_d1 << ",";
        std::cout << "\"abs_err_d2\":" << abs_err_d2 << ",";
        std::cout << "\"dy_dx_graph_requires_grad\":"
                  << (dy_dx_graph->requires_grad ? "true" : "false") << ",";
        std::cout << "\"g0_nodes\":" << g0.nodes << ",";
        std::cout << "\"g0_edges\":" << g0.edges << ",";
        std::cout << "\"g1_nodes\":" << g1.nodes << ",";
        std::cout << "\"g1_edges\":" << g1.edges;
        std::cout << "}\n";
        return 0;
    }

    std::cout << std::fixed << std::setprecision(12);
    std::cout << "Function: y = x^3 + sin(x) at x=" << x0 << '\n';
    std::cout << "Forward y           = " << y->value << '\n';
    std::cout << "dy/dx (detached)    = " << dy_dx_plain->value
              << " | expected=" << expected_d1
              << " | abs_err=" << abs_err_d1 << '\n';
    std::cout << "dy/dx (graph mode)  = " << dy_dx_graph->value
              << " | requires_grad=" << (dy_dx_graph->requires_grad ? "true" : "false") << '\n';
    std::cout << "d2y/dx2             = " << d2y_dx2->value
              << " | expected=" << expected_d2
              << " | abs_err=" << abs_err_d2 << '\n';
    std::cout << "Graph stats G0      = nodes=" << g0.nodes << ", edges=" << g0.edges << '\n';
    std::cout << "Graph stats G1      = nodes=" << g1.nodes << ", edges=" << g1.edges << '\n';

    if (print_graphs) {
        print_graph(y, "G0: forward graph y");
        print_graph(dy_dx_graph, "G1: first-derivative graph dy/dx (create_graph=true)");
    }

    return 0;
}
