#include "pinn/core/tensor.hpp"

#include "pinn/core/rng.hpp"

#include <cmath>
#include <cstring>
#include <numeric>

namespace pinn::core {

namespace {
constexpr int64_t kMaxRank = 8;

int64_t product_or_throw(const Tensor::Shape& shape) {
    if (shape.empty()) {
        return 1;
    }
    int64_t prod = 1;
    for (auto v : shape) {
        if (v < 0) {
            throw std::invalid_argument("shape dimension must be non-negative");
        }
        if (v == 0) {
            return 0;
        }
        if (prod > (std::numeric_limits<int64_t>::max() / v)) {
            throw std::overflow_error("numel overflow");
        }
        prod *= v;
    }
    return prod;
}

int64_t elem_size(DType dtype) {
    switch (dtype) {
        case DType::kFloat64:
            return static_cast<int64_t>(sizeof(double));
        case DType::kInt64:
            return static_cast<int64_t>(sizeof(int64_t));
        default:
            throw std::runtime_error("unsupported dtype");
    }
}

Tensor::Shape normalize_new_shape(const Tensor::Shape& old_shape, const Tensor::Shape& new_shape) {
    // Supports a single -1, like torch.
    int64_t infer_at = -1;
    Tensor::Shape out = new_shape;
    for (int64_t i = 0; i < static_cast<int64_t>(out.size()); ++i) {
        if (out[i] == -1) {
            if (infer_at >= 0) {
                throw std::invalid_argument("only one inferred dimension (-1) is supported");
            }
            infer_at = i;
        } else if (out[i] < 0) {
            throw std::invalid_argument("negative shape dimension");
        }
    }

    const int64_t old_numel = product_or_throw(old_shape);
    if (infer_at >= 0) {
        int64_t known = 1;
        for (auto v : out) {
            if (v == -1) {
                continue;
            }
            known *= v;
        }
        if (known == 0) {
            if (old_numel != 0) {
                throw std::invalid_argument("cannot infer shape with zero known product and nonzero numel");
            }
            out[infer_at] = 0;
        } else {
            if (old_numel % known != 0) {
                throw std::invalid_argument("reshape size mismatch");
            }
            out[infer_at] = old_numel / known;
        }
    }

    const int64_t new_numel = product_or_throw(out);
    if (new_numel != old_numel) {
        throw std::invalid_argument("reshape size mismatch");
    }

    return out;
}

}  // namespace

Tensor::Tensor(Shape shape, DType dtype) : dtype_{dtype}, shape_{std::move(shape)} {
    if (shape_.size() > kMaxRank) {
        throw std::invalid_argument("rank too large");
    }
    compute_strides_();
    const int64_t n = compute_numel_(shape_);
    buffer_.resize(static_cast<size_t>(n * elem_size(dtype_)));
    defined_ = true;
}

Tensor Tensor::scalar(double value) {
    Tensor t({1}, DType::kFloat64);
    t.data_ptr<double>()[0] = value;
    return t;
}

Tensor Tensor::scalar_int(int64_t value) {
    Tensor t({1}, DType::kInt64);
    t.data_ptr<int64_t>()[0] = value;
    return t;
}

Tensor Tensor::zeros(const Shape& shape, DType dtype) {
    Tensor t(shape, dtype);
    std::memset(t.buffer_.data(), 0, t.buffer_.size());
    return t;
}

Tensor Tensor::ones(const Shape& shape, DType dtype) {
    if (dtype == DType::kFloat64) {
        Tensor t(shape, dtype);
        auto* p = t.data_ptr<double>();
        const int64_t n = t.numel();
        for (int64_t i = 0; i < n; ++i) {
            p[i] = 1.0;
        }
        return t;
    }
    Tensor t(shape, dtype);
    auto* p = t.data_ptr<int64_t>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = 1;
    }
    return t;
}

Tensor Tensor::full(const Shape& shape, double value) {
    Tensor t(shape, DType::kFloat64);
    auto* p = t.data_ptr<double>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = value;
    }
    return t;
}

Tensor Tensor::full_int(const Shape& shape, int64_t value) {
    Tensor t(shape, DType::kInt64);
    auto* p = t.data_ptr<int64_t>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = value;
    }
    return t;
}

Tensor Tensor::zeros_like(const Tensor& other) {
    if (!other.defined()) {
        throw std::runtime_error("zeros_like requires defined tensor");
    }
    return zeros(other.shape(), other.dtype());
}

Tensor Tensor::ones_like(const Tensor& other) {
    if (!other.defined()) {
        throw std::runtime_error("ones_like requires defined tensor");
    }
    if (other.dtype() == DType::kFloat64) {
        return ones(other.shape(), DType::kFloat64);
    } else {
        return full_int(other.shape(), 1);
    }
}

Tensor Tensor::arange_int(int64_t start, int64_t end) {
    if (end < start) {
        throw std::invalid_argument("arange_int expects end>=start");
    }
    Tensor t({end - start}, DType::kInt64);
    auto* p = t.data_ptr<int64_t>();
    for (int64_t i = 0; i < (end - start); ++i) {
        p[i] = start + i;
    }
    return t;
}

Tensor Tensor::linspace(double start, double end, int64_t steps) {
    if (steps <= 0) {
        throw std::invalid_argument("linspace steps must be positive");
    }
    Tensor t({steps}, DType::kFloat64);
    auto* p = t.data_ptr<double>();
    if (steps == 1) {
        p[0] = start;
        return t;
    }
    const double step = (end - start) / static_cast<double>(steps - 1);
    for (int64_t i = 0; i < steps; ++i) {
        p[i] = start + step * static_cast<double>(i);
    }
    return t;
}

Tensor Tensor::rand_uniform(const Shape& shape, Rng& rng) {
    Tensor t(shape, DType::kFloat64);
    auto* p = t.data_ptr<double>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = rng.uniform01();
    }
    return t;
}

Tensor Tensor::randn(const Shape& shape, Rng& rng) {
    Tensor t(shape, DType::kFloat64);
    auto* p = t.data_ptr<double>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = rng.normal01();
    }
    return t;
}

Tensor Tensor::randint(int64_t low, int64_t high, const Shape& shape, Rng& rng) {
    if (high <= low) {
        throw std::invalid_argument("randint expects high>low");
    }
    Tensor t(shape, DType::kInt64);
    auto* p = t.data_ptr<int64_t>();
    const int64_t n = t.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] = rng.randint(low, high);
    }
    return t;
}

std::mt19937_64& Tensor::global_rng_(uint64_t seed) {
    static std::mt19937_64 rng{std::random_device{}()};
    if (seed != 0) {
        rng.seed(seed);
    }
    return rng;
}

Tensor Tensor::rand_uniform(const Shape& shape, uint64_t seed) {
    Rng rng_local(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed);
    return rand_uniform(shape, rng_local);
}

Tensor Tensor::randn(const Shape& shape, uint64_t seed) {
    Rng rng_local(seed == 0 ? 0x9e3779b97f4a7c15ULL : seed);
    return randn(shape, rng_local);
}

int64_t Tensor::size(int64_t d) const {
    check_defined_();
    if (d < 0 || d >= static_cast<int64_t>(shape_.size())) {
        throw std::out_of_range("size(dim) out of range");
    }
    return shape_[d];
}

int64_t Tensor::compute_numel_(const Shape& shape) {
    return product_or_throw(shape);
}

int64_t Tensor::numel() const noexcept {
    if (!defined_) {
        return 0;
    }
    return compute_numel_(shape_);
}

void Tensor::check_defined_() const {
    if (!defined_) {
        throw std::runtime_error("Tensor is undefined");
    }
}

void Tensor::compute_strides_() {
    strides_.assign(shape_.size(), 1);
    if (shape_.empty()) {
        return;
    }
    // Row-major contiguous
    int64_t stride = 1;
    for (int64_t i = static_cast<int64_t>(shape_.size()) - 1; i >= 0; --i) {
        strides_[static_cast<size_t>(i)] = stride;
        stride *= shape_[static_cast<size_t>(i)];
    }
}

int64_t Tensor::offset_1d_(int64_t i) const {
    if (shape_.size() != 1) {
        throw std::runtime_error("offset_1d requires 1D tensor");
    }
    if (i < 0 || i >= shape_[0]) {
        throw std::out_of_range("index out of range");
    }
    return i;
}

int64_t Tensor::offset_2d_(int64_t i, int64_t j) const {
    if (shape_.size() != 2) {
        throw std::runtime_error("offset_2d requires 2D tensor");
    }
    const auto rows = shape_[0];
    const auto cols = shape_[1];
    if (i < 0 || i >= rows || j < 0 || j >= cols) {
        throw std::out_of_range("index out of range");
    }
    // Row-major: i*cols + j
    return i * cols + j;
}

double Tensor::at(int64_t i) const {
    check_defined_();
    check_dtype_<double>();
    return data_ptr<double>()[offset_1d_(i)];
}

double Tensor::at(int64_t i, int64_t j) const {
    check_defined_();
    check_dtype_<double>();
    return data_ptr<double>()[offset_2d_(i, j)];
}

void Tensor::set(int64_t i, double v) {
    check_defined_();
    check_dtype_<double>();
    data_ptr<double>()[offset_1d_(i)] = v;
}

void Tensor::set(int64_t i, int64_t j, double v) {
    check_defined_();
    check_dtype_<double>();
    data_ptr<double>()[offset_2d_(i, j)] = v;
}

int64_t Tensor::at_int(int64_t i) const {
    check_defined_();
    check_dtype_<int64_t>();
    return data_ptr<int64_t>()[offset_1d_(i)];
}

void Tensor::set_int(int64_t i, int64_t v) {
    check_defined_();
    check_dtype_<int64_t>();
    data_ptr<int64_t>()[offset_1d_(i)] = v;
}

Tensor Tensor::clone() const {
    check_defined_();
    Tensor out(shape_, dtype_);
    out.buffer_ = buffer_;
    return out;
}

Tensor Tensor::reshape(const Shape& new_shape) const {
    check_defined_();
    Tensor out = *this;
    out.shape_ = normalize_new_shape(shape_, new_shape);
    out.compute_strides_();
    return out;
}

Tensor Tensor::unsqueeze(int64_t dim) const {
    check_defined_();
    Tensor::Shape out_shape = shape_;
    const int64_t rank = static_cast<int64_t>(out_shape.size());
    if (dim < 0) {
        dim = rank + 1 + dim;
    }
    if (dim < 0 || dim > rank) {
        throw std::out_of_range("unsqueeze dim out of range");
    }
    out_shape.insert(out_shape.begin() + dim, 1);
    return reshape(out_shape);
}

Tensor Tensor::squeeze(int64_t dim) const {
    check_defined_();
    Tensor::Shape out_shape;
    out_shape.reserve(shape_.size());

    if (dim == -1) {
        for (auto v : shape_) {
            if (v != 1) {
                out_shape.push_back(v);
            }
        }
        if (out_shape.empty()) {
            out_shape.push_back(1);
        }
        return reshape(out_shape);
    }

    int64_t d = dim;
    if (d < 0) {
        d = static_cast<int64_t>(shape_.size()) + d;
    }
    if (d < 0 || d >= static_cast<int64_t>(shape_.size())) {
        throw std::out_of_range("squeeze dim out of range");
    }

    for (int64_t i = 0; i < static_cast<int64_t>(shape_.size()); ++i) {
        if (i == d && shape_[static_cast<size_t>(i)] == 1) {
            continue;
        }
        out_shape.push_back(shape_[static_cast<size_t>(i)]);
    }
    if (out_shape.empty()) {
        out_shape.push_back(1);
    }
    return reshape(out_shape);
}

Tensor Tensor::transpose(int64_t dim0, int64_t dim1) const {
    check_defined_();
    if (dim() != 2) {
        throw std::runtime_error("transpose currently supports 2D only");
    }
    if (dim0 < 0) dim0 += 2;
    if (dim1 < 0) dim1 += 2;
    if (!((dim0 == 0 && dim1 == 1) || (dim0 == 1 && dim1 == 0))) {
        throw std::invalid_argument("transpose supports swapping dims 0 and 1 only");
    }
    if (dtype_ != DType::kFloat64) {
        throw std::runtime_error("transpose supports float64 only");
    }

    const int64_t rows = shape_[0];
    const int64_t cols = shape_[1];
    Tensor out({cols, rows}, dtype_);
    const auto* in = data_ptr<double>();
    auto* o = out.data_ptr<double>();

    for (int64_t i = 0; i < rows; ++i) {
        for (int64_t j = 0; j < cols; ++j) {
            o[j * rows + i] = in[i * cols + j];
        }
    }

    return out;
}

Tensor Tensor::matmul(const Tensor& other) const {
    check_defined_();
    other.check_defined_();
    if (dtype_ != DType::kFloat64 || other.dtype_ != DType::kFloat64) {
        throw std::runtime_error("matmul supports float64 only");
    }
    if (dim() != 2 || other.dim() != 2) {
        throw std::runtime_error("matmul supports 2D tensors only");
    }
    if (size(1) != other.size(0)) {
        throw std::runtime_error("matmul shape mismatch");
    }

    const int64_t M = size(0);
    const int64_t K = size(1);
    const int64_t N = other.size(1);

    Tensor out({M, N}, DType::kFloat64);
    const double* A = data_ptr<double>();
    const double* B = other.data_ptr<double>();
    double* C = out.data_ptr<double>();

    // Simple O(N^3) loop with OpenMP
#if defined(_OPENMP)
#pragma omp parallel for
#endif
    for (int64_t i = 0; i < M; ++i) {
        for (int64_t j = 0; j < N; ++j) {
            double sum = 0.0;
            for (int64_t k = 0; k < K; ++k) {
                sum += A[i * K + k] * B[k * N + j];
            }
            C[i * N + j] = sum;
        }
    }
    return out;
}

Tensor Tensor::slice(int64_t dim, int64_t start, int64_t end) const {
    check_defined_();
    if (dim < 0) {
        dim += this->dim();
    }
    if (dim < 0 || dim >= this->dim()) {
        throw std::out_of_range("slice dim out of range");
    }
    if (start < 0) {
        start += size(dim);
    }
    if (end < 0) {
        end += size(dim);
    }
    start = std::max<int64_t>(0, start);
    end = std::min<int64_t>(size(dim), end);
    if (end < start) {
        end = start;
    }

    // C-style: implement the few cases we need (dim=0 for 1D/2D/3D), contiguous output.
    if (dtype_ != DType::kFloat64 && dtype_ != DType::kInt64) {
        throw std::runtime_error("unsupported dtype for slice");
    }

    Shape out_shape = shape_;
    out_shape[static_cast<size_t>(dim)] = (end - start);
    Tensor out(out_shape, dtype_);

    const int64_t out_n = out.numel();
    if (out_n == 0) {
        return out;
    }

    // Only implement contiguous dim0 slicing for now.
    if (dim != 0) {
        throw std::runtime_error("slice currently supports dim=0 only");
    }

    const int64_t inner = numel() / size(0);
    const int64_t copy_elems = (end - start) * inner;

    const size_t bytes = static_cast<size_t>(copy_elems * elem_size(dtype_));
    const size_t src_off = static_cast<size_t>(start * inner * elem_size(dtype_));
    std::memcpy(out.buffer_.data(), buffer_.data() + src_off, bytes);
    (void)out_n;
    return out;
}

Tensor Tensor::to(DType dtype) const {
    check_defined_();
    if (dtype == dtype_) {
        return clone();
    }

    Tensor out(shape_, dtype);
    const int64_t n = numel();

    if (dtype_ == DType::kFloat64 && dtype == DType::kInt64) {
        const auto* in = data_ptr<double>();
        auto* o = out.data_ptr<int64_t>();
        for (int64_t i = 0; i < n; ++i) {
            o[i] = static_cast<int64_t>(in[i]);
        }
        return out;
    }

    if (dtype_ == DType::kInt64 && dtype == DType::kFloat64) {
        const auto* in = data_ptr<int64_t>();
        auto* o = out.data_ptr<double>();
        for (int64_t i = 0; i < n; ++i) {
            o[i] = static_cast<double>(in[i]);
        }
        return out;
    }

    throw std::runtime_error("unsupported dtype conversion");
}

static void ensure_same_shape_float(const Tensor& a, const Tensor& b) {
    if (a.dtype() != DType::kFloat64 || b.dtype() != DType::kFloat64) {
        throw std::runtime_error("expected float64 tensors");
    }
    if (a.shape() != b.shape()) {
        throw std::runtime_error("shape mismatch");
    }
}

Tensor Tensor::operator+(const Tensor& other) const {
    check_defined_();
    other.check_defined_();
    ensure_same_shape_float(*this, other);
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    const auto* b = other.data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] + b[i];
    }
    return out;
}

Tensor Tensor::operator-(const Tensor& other) const {
    check_defined_();
    other.check_defined_();
    ensure_same_shape_float(*this, other);
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    const auto* b = other.data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] - b[i];
    }
    return out;
}

Tensor Tensor::operator*(const Tensor& other) const {
    check_defined_();
    other.check_defined_();
    ensure_same_shape_float(*this, other);
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    const auto* b = other.data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] * b[i];
    }
    return out;
}

Tensor Tensor::operator/(const Tensor& other) const {
    check_defined_();
    other.check_defined_();
    ensure_same_shape_float(*this, other);
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    const auto* b = other.data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] / b[i];
    }
    return out;
}

Tensor Tensor::operator+(double s) const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] + s;
    }
    return out;
}

Tensor Tensor::operator-(double s) const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] - s;
    }
    return out;
}

Tensor Tensor::operator*(double s) const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] * s;
    }
    return out;
}

Tensor Tensor::operator/(double s) const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] / s;
    }
    return out;
}

Tensor Tensor::abs() const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = std::abs(a[i]);
    }
    return out;
}

Tensor Tensor::pow(double exponent) const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = std::pow(a[i], exponent);
    }
    return out;
}

Tensor Tensor::sqrt() const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = std::sqrt(a[i]);
    }
    return out;
}

Tensor Tensor::sin() const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = std::sin(a[i]);
    }
    return out;
}

Tensor Tensor::tanh() const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = std::tanh(a[i]);
    }
    return out;
}

Tensor Tensor::relu() const {
    check_defined_();
    check_dtype_<double>();
    Tensor out(shape_, DType::kFloat64);
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        o[i] = a[i] > 0.0 ? a[i] : 0.0;
    }
    return out;
}

Tensor Tensor::sum_all() const {
    check_defined_();
    check_dtype_<double>();
    const int64_t n = numel();
    const auto* a = data_ptr<double>();
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        s += a[i];
    }
    return Tensor::scalar(s);
}

Tensor Tensor::mean_all() const {
    check_defined_();
    check_dtype_<double>();
    const int64_t n = numel();
    if (n == 0) {
        return Tensor::scalar(0.0);
    }
    auto s = sum_all().item<double>();
    return Tensor::scalar(s / static_cast<double>(n));
}

Tensor Tensor::sum(int64_t dim, bool keepdim) const {
    check_defined_();
    check_dtype_<double>();
    if (dim < 0) {
        dim += this->dim();
    }
    if (dim < 0 || dim >= this->dim()) {
        throw std::out_of_range("sum dim out of range");
    }
    if (this->dim() == 1) {
        if (dim != 0) {
            throw std::out_of_range("sum dim out of range");
        }
        return sum_all();
    }
    if (this->dim() != 2) {
        throw std::runtime_error("sum currently supports 1D/2D only");
    }

    const int64_t rows = shape_[0];
    const int64_t cols = shape_[1];
    const auto* a = data_ptr<double>();

    if (dim == 0) {
        // sum over rows -> [cols]
        Tensor out(keepdim ? Shape{1, cols} : Shape{cols}, DType::kFloat64);
        auto* o = out.data_ptr<double>();
        for (int64_t j = 0; j < cols; ++j) {
            double s = 0.0;
            for (int64_t i = 0; i < rows; ++i) {
                s += a[i * cols + j];
            }
            o[keepdim ? j : j] = s;
        }
        return out;
    }

    // dim==1 sum over cols -> [rows]
    Tensor out(keepdim ? Shape{rows, 1} : Shape{rows}, DType::kFloat64);
    auto* o = out.data_ptr<double>();
    for (int64_t i = 0; i < rows; ++i) {
        double s = 0.0;
        for (int64_t j = 0; j < cols; ++j) {
            s += a[i * cols + j];
        }
        o[i] = s;
    }
    return out;
}

Tensor Tensor::mean(int64_t dim, bool keepdim) const {
    check_defined_();
    check_dtype_<double>();
    Tensor s = sum(dim, keepdim);

    int64_t denom = size(dim < 0 ? dim + this->dim() : dim);
    if (denom == 0) {
        return zeros(s.shape(), s.dtype());
    }

    auto* p = s.data_ptr<double>();
    const int64_t n = s.numel();
    for (int64_t i = 0; i < n; ++i) {
        p[i] /= static_cast<double>(denom);
    }
    return s;
}

Tensor Tensor::norm(int64_t p, int64_t dim, bool keepdim) const {
    check_defined_();
    check_dtype_<double>();
    if (p != 2) {
        throw std::runtime_error("norm currently supports p=2 only");
    }
    if (dim == -1) {
        // L2 norm of all elements
        const int64_t n = numel();
        const auto* a = data_ptr<double>();
        double s = 0.0;
        for (int64_t i = 0; i < n; ++i) {
            s += a[i] * a[i];
        }
        return Tensor::scalar(std::sqrt(s));
    }

    if (this->dim() != 2) {
        throw std::runtime_error("norm(dim) currently supports 2D only");
    }
    if (dim < 0) dim += 2;
    if (dim != 1 && dim != 0) {
        throw std::out_of_range("norm dim out of range");
    }

    const int64_t rows = shape_[0];
    const int64_t cols = shape_[1];
    const auto* a = data_ptr<double>();

    if (dim == 1) {
        Tensor out(keepdim ? Shape{rows, 1} : Shape{rows}, DType::kFloat64);
        auto* o = out.data_ptr<double>();
        for (int64_t i = 0; i < rows; ++i) {
            double s = 0.0;
            for (int64_t j = 0; j < cols; ++j) {
                const double v = a[i * cols + j];
                s += v * v;
            }
            o[i] = std::sqrt(s);
        }
        return out;
    }

    Tensor out(keepdim ? Shape{1, cols} : Shape{cols}, DType::kFloat64);
    auto* o = out.data_ptr<double>();
    for (int64_t j = 0; j < cols; ++j) {
        double s = 0.0;
        for (int64_t i = 0; i < rows; ++i) {
            const double v = a[i * cols + j];
            s += v * v;
        }
        o[j] = std::sqrt(s);
    }
    return out;
}

std::pair<Tensor, Tensor> Tensor::topk(int64_t k, int64_t dim, bool largest) const {
    check_defined_();
    check_dtype_<double>();
    if (dim < 0) {
        dim += this->dim();
    }
    if (this->dim() != 1 || dim != 0) {
        throw std::runtime_error("topk currently supports 1D dim=0 only");
    }
    const int64_t n = shape_[0];
    if (k < 0 || k > n) {
        throw std::out_of_range("topk k out of range");
    }

    std::vector<std::pair<double, int64_t>> items;
    items.reserve(static_cast<size_t>(n));
    const auto* a = data_ptr<double>();
    for (int64_t i = 0; i < n; ++i) {
        items.emplace_back(a[i], i);
    }

    auto cmp = [largest](const auto& lhs, const auto& rhs) {
        return largest ? (lhs.first > rhs.first) : (lhs.first < rhs.first);
    };

    if (k < n) {
        std::nth_element(items.begin(), items.begin() + k, items.end(), cmp);
        items.resize(static_cast<size_t>(k));
    }
    std::sort(items.begin(), items.end(), cmp);

    Tensor values({k}, DType::kFloat64);
    Tensor indices({k}, DType::kInt64);
    auto* v = values.data_ptr<double>();
    auto* idx = indices.data_ptr<int64_t>();
    for (int64_t i = 0; i < k; ++i) {
        v[i] = items[static_cast<size_t>(i)].first;
        idx[i] = items[static_cast<size_t>(i)].second;
    }

    return {values, indices};
}

Tensor Tensor::cat(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        return {};
    }
    tensors[0].check_defined_();
    const auto dtype = tensors[0].dtype();
    for (const auto& t : tensors) {
        t.check_defined_();
        if (t.dtype() != dtype) {
            throw std::runtime_error("cat dtype mismatch");
        }
    }

    if (dim < 0) {
        dim += tensors[0].dim();
    }
    if (dim != 0) {
        throw std::runtime_error("cat currently supports dim=0 only");
    }

    // Support 1D and 2D (batch concat)
    if (tensors[0].dim() != 1 && tensors[0].dim() != 2) {
        throw std::runtime_error("cat supports 1D/2D only");
    }

    Tensor::Shape out_shape = tensors[0].shape();
    int64_t total0 = 0;
    for (const auto& t : tensors) {
        if (t.dim() != tensors[0].dim()) {
            throw std::runtime_error("cat rank mismatch");
        }
        for (int64_t d = 1; d < t.dim(); ++d) {
            if (t.size(d) != out_shape[static_cast<size_t>(d)]) {
                throw std::runtime_error("cat shape mismatch (non-concat dims)");
            }
        }
        total0 += t.size(0);
    }
    out_shape[0] = total0;

    Tensor out(out_shape, dtype);

    const int64_t inner = tensors[0].numel() / tensors[0].size(0);
    size_t dst_off = 0;
    const size_t elem_bytes = static_cast<size_t>(elem_size(dtype));
    for (const auto& t : tensors) {
        const size_t bytes = static_cast<size_t>(t.size(0) * inner) * elem_bytes;
        std::memcpy(out.buffer_.data() + dst_off, t.buffer_.data(), bytes);
        dst_off += bytes;
    }

    return out;
}

Tensor Tensor::stack(const std::vector<Tensor>& tensors, int64_t dim) {
    if (tensors.empty()) {
        return {};
    }
    if (dim != 0) {
        throw std::runtime_error("stack currently supports dim=0 only");
    }
    // Stack adds a new leading dim: [N, ...]
    Tensor::Shape base = tensors[0].shape();
    const auto dtype = tensors[0].dtype();
    for (const auto& t : tensors) {
        if (!t.defined()) {
            throw std::runtime_error("stack tensor undefined");
        }
        if (t.dtype() != dtype || t.shape() != base) {
            throw std::runtime_error("stack requires same dtype and shape");
        }
    }
    Tensor::Shape out_shape;
    out_shape.reserve(base.size() + 1);
    out_shape.push_back(static_cast<int64_t>(tensors.size()));
    out_shape.insert(out_shape.end(), base.begin(), base.end());

    Tensor out(out_shape, dtype);
    const size_t bytes_each = static_cast<size_t>(tensors[0].numel() * elem_size(dtype));
    for (size_t i = 0; i < tensors.size(); ++i) {
        std::memcpy(out.buffer_.data() + i * bytes_each, tensors[i].buffer_.data(), bytes_each);
    }

    return out;
}

Tensor mse_loss(const Tensor& input, const Tensor& target) {
    if (!input.defined() || !target.defined()) {
        throw std::runtime_error("mse_loss requires defined tensors");
    }
    ensure_same_shape_float(input, target);
    const int64_t n = input.numel();
    if (n == 0) {
        return Tensor::scalar(0.0);
    }
    const auto* a = input.data_ptr<double>();
    const auto* b = target.data_ptr<double>();
    double s = 0.0;
    for (int64_t i = 0; i < n; ++i) {
        const double d = a[i] - b[i];
        s += d * d;
    }
    return Tensor::scalar(s / static_cast<double>(n));
}

}  // namespace pinn::core
