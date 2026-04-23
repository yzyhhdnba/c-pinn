#include <algorithm>
#include <cmath>
#include <functional>
#include <iomanip>
#include <iostream>
#include <memory>
#include <random>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
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

NodePtr grad(const NodePtr& output, const NodePtr& input, bool create_graph) {
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

int sgd_step(const std::vector<NodePtr>& params, double lr, double grad_clip) {
    int nan_grad_count = 0;
    for (const auto& p : params) {
        if (!p || !p->is_leaf || !p->requires_grad) {
            continue;
        }
        if (!std::isfinite(p->grad)) {
            ++nan_grad_count;
            continue;
        }
        double gv = std::clamp(p->grad, -grad_clip, grad_clip);
        p->value -= lr * gv;
    }
    return nan_grad_count;
}

struct VarMatrix {
    int rows = 0;
    int cols = 0;
    std::vector<NodePtr> data;

    VarMatrix() = default;

    VarMatrix(int r, int c) : rows(r), cols(c), data(static_cast<std::size_t>(r * c), constant(0.0)) {}

    NodePtr& operator()(int r, int c) {
        return data[static_cast<std::size_t>(r * cols + c)];
    }

    const NodePtr& operator()(int r, int c) const {
        return data[static_cast<std::size_t>(r * cols + c)];
    }
};

VarMatrix mat_add(const VarMatrix& a, const VarMatrix& b) {
    if (a.rows != b.rows || a.cols != b.cols) {
        throw std::runtime_error("mat_add shape mismatch");
    }
    VarMatrix out(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            out(i, j) = add(a(i, j), b(i, j));
        }
    }
    return out;
}

VarMatrix mat_add_rowwise(const VarMatrix& a, const VarMatrix& bias) {
    if (bias.rows != 1 || bias.cols != a.cols) {
        throw std::runtime_error("mat_add_rowwise shape mismatch");
    }
    VarMatrix out(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            out(i, j) = add(a(i, j), bias(0, j));
        }
    }
    return out;
}

VarMatrix mat_tanh(const VarMatrix& a) {
    VarMatrix out(a.rows, a.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < a.cols; ++j) {
            out(i, j) = tanh(a(i, j));
        }
    }
    return out;
}

VarMatrix matmul(const VarMatrix& a, const VarMatrix& b) {
    if (a.cols != b.rows) {
        throw std::runtime_error("matmul shape mismatch");
    }
    VarMatrix out(a.rows, b.cols);
    for (int i = 0; i < a.rows; ++i) {
        for (int j = 0; j < b.cols; ++j) {
            NodePtr acc = constant(0.0);
            for (int k = 0; k < a.cols; ++k) {
                acc = add(acc, mul(a(i, k), b(k, j)));
            }
            out(i, j) = acc;
        }
    }
    return out;
}

struct TinyFNN {
    std::vector<VarMatrix> weights;
    std::vector<VarMatrix> biases;

    explicit TinyFNN(const std::vector<int>& layers, std::uint32_t seed) {
        if (layers.size() < 2) {
            throw std::runtime_error("layers size must be >= 2");
        }

        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> dist(-1.0, 1.0);

        for (std::size_t i = 0; i + 1 < layers.size(); ++i) {
            int in = layers[i];
            int out = layers[i + 1];
            double scale = std::sqrt(2.0 / static_cast<double>(in + out));

            VarMatrix w(in, out);
            VarMatrix b(1, out);
            for (int r = 0; r < in; ++r) {
                for (int c = 0; c < out; ++c) {
                    w(r, c) = variable(dist(rng) * scale, true);
                }
            }
            for (int c = 0; c < out; ++c) {
                b(0, c) = variable(0.0, true);
            }
            weights.push_back(std::move(w));
            biases.push_back(std::move(b));
        }
    }

    NodePtr forward_scalar(const NodePtr& x, const NodePtr& t) const {
        VarMatrix a(1, 2);
        a(0, 0) = x;
        a(0, 1) = t;

        for (std::size_t i = 0; i < weights.size(); ++i) {
            auto z = mat_add_rowwise(matmul(a, weights[i]), biases[i]);
            if (i + 1 < weights.size()) {
                a = mat_tanh(z);
            } else {
                a = z;
            }
        }
        return a(0, 0);
    }

    std::vector<NodePtr> parameters() const {
        std::vector<NodePtr> params;
        for (const auto& w : weights) {
            for (const auto& v : w.data) {
                params.push_back(v);
            }
        }
        for (const auto& b : biases) {
            for (const auto& v : b.data) {
                params.push_back(v);
            }
        }
        return params;
    }
};

enum class EquationKind {
    KdV,
    SineGordon,
    AllenCahn,
};

std::string equation_name(EquationKind kind) {
    switch (kind) {
        case EquationKind::KdV:
            return "KdV";
        case EquationKind::SineGordon:
            return "Sine-Gordon";
        case EquationKind::AllenCahn:
            return "Allen-Cahn";
    }
    return "Unknown";
}

NodePtr residual_at_point(const TinyFNN& net,
                          EquationKind kind,
                          double x0,
                          double t0,
                          bool create_graph) {
    auto x = variable(x0, true);
    auto t = variable(t0, true);
    auto u = net.forward_scalar(x, t);

    auto u_t = grad(u, t, create_graph);
    auto u_x = grad(u, x, create_graph);
    auto u_xx = grad(u_x, x, create_graph);

    if (kind == EquationKind::KdV) {
        auto u_xxx = grad(u_xx, x, create_graph);
        return add(add(u_t, mul(constant(6.0), mul(u, u_x))), u_xxx);
    }

    if (kind == EquationKind::SineGordon) {
        auto u_tt = grad(u_t, t, create_graph);
        return add(sub(u_tt, u_xx), sin(u));
    }

    auto u3 = mul(u, mul(u, u));
    auto diffusion = mul(constant(1.0e-4), u_xx);
    auto reaction = mul(constant(5.0), sub(u3, u));
    return add(sub(u_t, diffusion), reaction);
}

NodePtr loss_on_points(const TinyFNN& net,
                       EquationKind kind,
                       const std::vector<std::pair<double, double>>& pts,
                       bool create_graph) {
    NodePtr sum = constant(0.0);
    for (const auto& p : pts) {
        auto r = residual_at_point(net, kind, p.first, p.second, create_graph);
        sum = add(sum, mul(r, r));
    }
    return mul(sum, constant(1.0 / static_cast<double>(pts.size())));
}

struct TrainStats {
    std::string equation;
    int iters = 0;
    int samples = 0;
    int params = 0;
    double initial_loss = 0.0;
    double final_loss = 0.0;
    double best_loss = 0.0;
    bool loss_decreased = false;
    int nan_grad_count = 0;
};

struct SelfCheckStats {
    bool grad_buffer_matches_direct = false;
    bool zero_grad_clears_buffers = false;
    bool sgd_step_updates_parameter = false;
    bool backward_accumulates_grad = false;
    double direct_grad = 0.0;
    double buffer_grad = 0.0;
    double accumulated_grad = 0.0;
    double value_before_step = 0.0;
    double value_after_step = 0.0;
};

SelfCheckStats run_self_checks() {
    SelfCheckStats s;

    auto p = variable(2.0, true);
    auto loss = mul(p, p);
    const double direct_grad = grad(loss, p, false)->value;

    zero_grad({p});
    backward(loss, {p});
    const double buffer_grad = p->grad;

    s.direct_grad = direct_grad;
    s.buffer_grad = buffer_grad;
    s.grad_buffer_matches_direct = std::abs(buffer_grad - direct_grad) < 1.0e-12;

    backward(loss, {p});
    s.accumulated_grad = p->grad;
    s.backward_accumulates_grad = std::abs(p->grad - 2.0 * direct_grad) < 1.0e-12;

    zero_grad({p});
    s.zero_grad_clears_buffers = std::abs(p->grad) < 1.0e-12;

    auto q = variable(1.5, true);
    auto q_loss = mul(q, q);
    zero_grad({q});
    backward(q_loss, {q});
    s.value_before_step = q->value;
    sgd_step({q}, 0.1, 100.0);
    s.value_after_step = q->value;
    s.sgd_step_updates_parameter = s.value_after_step < s.value_before_step;

    return s;
}

TrainStats train_equation(EquationKind kind,
                          int iters,
                          int samples,
                          std::uint32_t seed,
                          double lr,
                          double grad_clip) {
    TinyFNN net({2, 4, 1}, seed);
    auto params = net.parameters();

    std::mt19937 rng(seed + 1001U);
    std::uniform_real_distribution<double> dist(0.0, 1.0);

    std::vector<std::pair<double, double>> points;
    points.reserve(static_cast<std::size_t>(samples));
    for (int i = 0; i < samples; ++i) {
        points.emplace_back(dist(rng), dist(rng));
    }

    auto init_loss_node = loss_on_points(net, kind, points, false);
    double init_loss = init_loss_node->value;
    double best_loss = init_loss;
    int nan_grad_count = 0;

    for (int iter = 0; iter < iters; ++iter) {
        auto loss = loss_on_points(net, kind, points, true);
        zero_grad(params);
        backward(loss, params);
        nan_grad_count += sgd_step(params, lr, grad_clip);

        auto eval_loss = loss_on_points(net, kind, points, false);
        best_loss = std::min(best_loss, eval_loss->value);
    }

    auto final_loss_node = loss_on_points(net, kind, points, false);
    double final_loss = final_loss_node->value;

    TrainStats s;
    s.equation = equation_name(kind);
    s.iters = iters;
    s.samples = samples;
    s.params = static_cast<int>(params.size());
    s.initial_loss = init_loss;
    s.final_loss = final_loss;
    s.best_loss = best_loss;
    s.loss_decreased = final_loss < init_loss;
    s.nan_grad_count = nan_grad_count;
    return s;
}

bool parse_kind(const std::string& s, EquationKind* kind) {
    if (s == "kdv") {
        *kind = EquationKind::KdV;
        return true;
    }
    if (s == "sine_gordon" || s == "sg") {
        *kind = EquationKind::SineGordon;
        return true;
    }
    if (s == "allen_cahn" || s == "ac") {
        *kind = EquationKind::AllenCahn;
        return true;
    }
    return false;
}

void print_json(const std::vector<TrainStats>& stats,
                int iters,
                int samples,
                std::uint32_t seed,
                double lr,
                double grad_clip,
                const SelfCheckStats& self_check) {
    std::cout << std::fixed << std::setprecision(15);
    std::cout << "{";
    std::cout << "\"config\":{";
    std::cout << "\"iters\":" << iters << ",";
    std::cout << "\"samples\":" << samples << ",";
    std::cout << "\"seed\":" << seed << ",";
    std::cout << "\"lr\":" << lr << ",";
    std::cout << "\"grad_clip\":" << grad_clip;
    std::cout << "},";
    std::cout << "\"self_check\":{";
    std::cout << "\"grad_buffer_matches_direct\":"
              << (self_check.grad_buffer_matches_direct ? "true" : "false") << ",";
    std::cout << "\"zero_grad_clears_buffers\":"
              << (self_check.zero_grad_clears_buffers ? "true" : "false") << ",";
    std::cout << "\"sgd_step_updates_parameter\":"
              << (self_check.sgd_step_updates_parameter ? "true" : "false") << ",";
    std::cout << "\"backward_accumulates_grad\":"
              << (self_check.backward_accumulates_grad ? "true" : "false") << ",";
    std::cout << "\"direct_grad\":" << self_check.direct_grad << ",";
    std::cout << "\"buffer_grad\":" << self_check.buffer_grad << ",";
    std::cout << "\"accumulated_grad\":" << self_check.accumulated_grad << ",";
    std::cout << "\"value_before_step\":" << self_check.value_before_step << ",";
    std::cout << "\"value_after_step\":" << self_check.value_after_step;
    std::cout << "},";
    std::cout << "\"results\":[";
    for (std::size_t i = 0; i < stats.size(); ++i) {
        const auto& s = stats[i];
        std::cout << "{";
        std::cout << "\"equation\":\"" << s.equation << "\",";
        std::cout << "\"iters\":" << s.iters << ",";
        std::cout << "\"samples\":" << s.samples << ",";
        std::cout << "\"params\":" << s.params << ",";
        std::cout << "\"initial_loss\":" << s.initial_loss << ",";
        std::cout << "\"final_loss\":" << s.final_loss << ",";
        std::cout << "\"best_loss\":" << s.best_loss << ",";
        std::cout << "\"loss_ratio\":" << (s.final_loss / std::max(1.0e-300, s.initial_loss)) << ",";
        std::cout << "\"loss_decreased\":" << (s.loss_decreased ? "true" : "false") << ",";
        std::cout << "\"nan_grad_count\":" << s.nan_grad_count;
        std::cout << "}";
        if (i + 1 < stats.size()) {
            std::cout << ",";
        }
    }
    std::cout << "]";
    std::cout << "}\n";
}

}  // namespace mini_ad

int main(int argc, char** argv) {
    using namespace mini_ad;

    bool json_mode = false;
    std::string eq_arg = "all";
    int iters = 25;
    int samples = 6;
    std::uint32_t seed = 7U;
    double lr = 5.0e-4;
    double grad_clip = 50.0;
    bool self_check_only = false;

    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--json") {
            json_mode = true;
            continue;
        }
        if (arg == "--self-check-only") {
            self_check_only = true;
            continue;
        }
        if (arg == "--equation" && i + 1 < argc) {
            eq_arg = argv[++i];
            continue;
        }
        if (arg == "--iters" && i + 1 < argc) {
            iters = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--samples" && i + 1 < argc) {
            samples = std::stoi(argv[++i]);
            continue;
        }
        if (arg == "--seed" && i + 1 < argc) {
            seed = static_cast<std::uint32_t>(std::stoul(argv[++i]));
            continue;
        }
        if (arg == "--lr" && i + 1 < argc) {
            lr = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--grad-clip" && i + 1 < argc) {
            grad_clip = std::stod(argv[++i]);
            continue;
        }
        if (arg == "--help" || arg == "-h") {
            std::cout << "Usage: example_autodiff_matrix_graph [--json] [--equation all|kdv|sine_gordon|allen_cahn] "
                         "[--iters N] [--samples N] [--seed N] [--lr X] [--grad-clip X] [--self-check-only]\n";
            return 0;
        }
        std::cerr << "Unknown arg: " << arg << "\n";
        return 2;
    }

    std::vector<EquationKind> run_list;
    if (eq_arg == "all") {
        run_list = {EquationKind::KdV, EquationKind::SineGordon, EquationKind::AllenCahn};
    } else {
        EquationKind kind;
        if (!parse_kind(eq_arg, &kind)) {
            std::cerr << "Invalid equation: " << eq_arg << "\n";
            return 2;
        }
        run_list.push_back(kind);
    }

    const auto self_check = run_self_checks();

    std::vector<TrainStats> stats;
    if (!self_check_only) {
        for (std::size_t i = 0; i < run_list.size(); ++i) {
            stats.push_back(train_equation(
                run_list[i],
                iters,
                samples,
                seed + static_cast<std::uint32_t>(100 * i),
                lr,
                grad_clip));
        }
    }

    if (json_mode) {
        print_json(stats, iters, samples, seed, lr, grad_clip, self_check);
        return 0;
    }

    std::cout << std::fixed << std::setprecision(9);
    std::cout << "Matrix AD mini-PINN prototype (with matmul)\n";
    std::cout << "config: iters=" << iters
              << " samples=" << samples
              << " seed=" << seed
              << " lr=" << lr
              << " grad_clip=" << grad_clip << "\n\n";

    std::cout << "[self-check] grad_buffer_matches_direct="
              << (self_check.grad_buffer_matches_direct ? "true" : "false")
              << " zero_grad_clears_buffers="
              << (self_check.zero_grad_clears_buffers ? "true" : "false")
              << " backward_accumulates_grad="
              << (self_check.backward_accumulates_grad ? "true" : "false")
              << " sgd_step_updates_parameter="
              << (self_check.sgd_step_updates_parameter ? "true" : "false")
              << "\n\n";

    if (self_check_only) {
        return 0;
    }

    bool all_decreased = true;
    for (const auto& s : stats) {
        std::cout << "[" << s.equation << "] "
                  << "params=" << s.params
                  << " init_loss=" << s.initial_loss
                  << " final_loss=" << s.final_loss
                  << " best_loss=" << s.best_loss
                  << " ratio=" << (s.final_loss / std::max(1.0e-300, s.initial_loss))
                  << " decreased=" << (s.loss_decreased ? "true" : "false")
                  << " nan_grad=" << s.nan_grad_count
                  << "\n";
        if (!s.loss_decreased) {
            all_decreased = false;
        }
    }

    if (!all_decreased) {
        std::cout << "\nWarning: at least one equation did not decrease final loss under this tiny config.\n";
    }
    return 0;
}
