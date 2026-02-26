#pragma once

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <limits>
#include <random>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <utility>
#include <vector>

namespace pinn::core {

class Rng;

enum class DType {
    kFloat64,
    kInt64,
};

class Tensor {
  public:
    using Shape = std::vector<int64_t>;

    Tensor() = default;

    explicit Tensor(Shape shape, DType dtype = DType::kFloat64);

    static Tensor scalar(double value);
    static Tensor scalar_int(int64_t value);

    static Tensor zeros(const Shape& shape, DType dtype = DType::kFloat64);
    static Tensor ones(const Shape& shape, DType dtype = DType::kFloat64);
    static Tensor full(const Shape& shape, double value);
    static Tensor full_int(const Shape& shape, int64_t value);

    static Tensor zeros_like(const Tensor& other);
    static Tensor ones_like(const Tensor& other);

    static Tensor arange_int(int64_t start, int64_t end);
    static Tensor linspace(double start, double end, int64_t steps);

    static Tensor rand_uniform(const Shape& shape, Rng& rng);
    static Tensor randn(const Shape& shape, Rng& rng);
    static Tensor randint(int64_t low, int64_t high, const Shape& shape, Rng& rng);

    // Convenience wrappers (seed!=0 reseeds an internal rng)
    static Tensor rand_uniform(const Shape& shape, uint64_t seed = 0);
    static Tensor randn(const Shape& shape, uint64_t seed = 0);

    bool defined() const noexcept { return defined_; }
    bool empty() const noexcept { return !defined_ || numel() == 0; }

    DType dtype() const noexcept { return dtype_; }
    const Shape& shape() const noexcept { return shape_; }
    int64_t dim() const noexcept { return static_cast<int64_t>(shape_.size()); }
    int64_t size(int64_t d) const;
    int64_t numel() const noexcept;

    Tensor clone() const;

    template <typename T>
    T item() const {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int64_t>, "item<T> supports only double or int64_t");
        if (!defined_) {
            throw std::runtime_error("Tensor is undefined");
        }
        if (numel() != 1) {
            throw std::runtime_error("item<T>() requires a single element tensor");
        }
        if constexpr (std::is_same_v<T, double>) {
            if (dtype_ != DType::kFloat64) {
                throw std::runtime_error("item<double>() called on non-float tensor");
            }
            return *data_ptr<double>();
        } else {
            if (dtype_ != DType::kInt64) {
                throw std::runtime_error("item<int64_t>() called on non-int tensor");
            }
            return *data_ptr<int64_t>();
        }
    }

    template <typename T>
    T* data_ptr() {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int64_t>, "data_ptr<T> supports only double or int64_t");
        check_defined_();
        check_dtype_<T>();
        return reinterpret_cast<T*>(buffer_.data());
    }

    template <typename T>
    const T* data_ptr() const {
        static_assert(std::is_same_v<T, double> || std::is_same_v<T, int64_t>, "data_ptr<T> supports only double or int64_t");
        check_defined_();
        check_dtype_<T>();
        return reinterpret_cast<const T*>(buffer_.data());
    }

    // 1D/2D element access (bounds-checked)
    double at(int64_t i) const;
    double at(int64_t i, int64_t j) const;
    void set(int64_t i, double v);
    void set(int64_t i, int64_t j, double v);

    int64_t at_int(int64_t i) const;
    void set_int(int64_t i, int64_t v);

    // Shape ops (return new contiguous tensors)
    Tensor reshape(const Shape& new_shape) const;
    Tensor view(const Shape& new_shape) const { return reshape(new_shape); }
    Tensor unsqueeze(int64_t dim) const;
    Tensor squeeze(int64_t dim = -1) const;
    Tensor transpose(int64_t dim0, int64_t dim1) const;
    Tensor slice(int64_t dim, int64_t start, int64_t end) const;

    // Type conversion
    Tensor to(DType dtype) const;

    // Elementwise ops (float only)
    Tensor operator+(const Tensor& other) const;
    Tensor operator-(const Tensor& other) const;
    Tensor operator*(const Tensor& other) const;
    Tensor operator/(const Tensor& other) const;

    Tensor operator+(double s) const;
    Tensor operator-(double s) const;
    Tensor operator*(double s) const;
    Tensor operator/(double s) const;

    // Matrix multiplication
    Tensor matmul(const Tensor& other) const;

    // Math
    Tensor abs() const;
    Tensor pow(double exponent) const;
    Tensor sqrt() const;
    Tensor sin() const;
    Tensor tanh() const;
    Tensor relu() const;

    // Reductions
    Tensor sum_all() const;
    Tensor mean_all() const;
    Tensor sum(int64_t dim, bool keepdim = false) const;
    Tensor mean(int64_t dim, bool keepdim = false) const;
    Tensor norm(int64_t p = 2, int64_t dim = -1, bool keepdim = false) const;

    // Selection
    std::pair<Tensor, Tensor> topk(int64_t k, int64_t dim = 0, bool largest = true) const;

    // Concatenation
    static Tensor cat(const std::vector<Tensor>& tensors, int64_t dim = 0);
    static Tensor stack(const std::vector<Tensor>& tensors, int64_t dim = 0);

  private:
    DType dtype_{DType::kFloat64};
    Shape shape_{};
    std::vector<int64_t> strides_{};
    std::vector<std::byte> buffer_{};
    bool defined_{false};

    void check_defined_() const;
    static int64_t compute_numel_(const Shape& shape);
    void compute_strides_();

    template <typename T>
    void check_dtype_() const {
        if constexpr (std::is_same_v<T, double>) {
            if (dtype_ != DType::kFloat64) {
                throw std::runtime_error("Tensor dtype mismatch: expected float64");
            }
        } else {
            if (dtype_ != DType::kInt64) {
                throw std::runtime_error("Tensor dtype mismatch: expected int64");
            }
        }
    }

    int64_t offset_1d_(int64_t i) const;
    int64_t offset_2d_(int64_t i, int64_t j) const;

    static std::mt19937_64& global_rng_(uint64_t seed);
};

// Free helpers matching current code style later
Tensor mse_loss(const Tensor& input, const Tensor& target);

}  // namespace pinn::core
