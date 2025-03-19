#ifndef UTILS_H
#define UTILS_H

#include <iostream>
#include <iterator>
#include <format>
#include <vector>
#include <string>
#include <tuple>

class range {
  int start, stop, step;
public:

  class iterator {
    int value, step;
  public:
    iterator(int start, int step) : value(start), step(step) {}

    int operator*() const { return value; }
    iterator& operator++() { value += step; return *this; }
    
    bool operator!=(const iterator& other) const {
        return (step > 0) ? (value < other.value) : (value > other.value);
    }
  };

  range(int stop) : start(0), stop(stop), step(1) {}
  range(int start, int stop, int step = 1) : start(start), stop(stop), step(step) {}

  iterator begin() const { return { start, step }; }
  iterator end() const { return { stop, step }; }
};

template<typename T>
class enumerate {
    T& container;
    int start;
public:
  class iterator {
    typename T::iterator iter, end;
    int index;
  public:
    iterator(T& container, int start):
      iter(container.begin()), end(container.end()), index(start) {}

    bool operator!=(const iterator& other) const {
      return iter != other.iter;
    }

    void operator++() { ++iter; ++index; }

    std::pair<int, typename T::value_type> operator*() const {
      return { index, *iter };
    }
  };

  enumerate(T& container, int start = 0) : container(container), start(start) {}

  auto begin() { return iterator(container, start); }
  auto end() { return iterator(container, start + container.size()); }
};

template <typename... Containers>
class zip {
private:
  std::tuple<Containers&...> containers;

  template <std::size_t... Is>
  auto make_begin_tuple(std::index_sequence<Is...>) {
    return std::tuple(std::begin(std::get<Is>(containers))...);
  }

  template <std::size_t... Is>
  auto make_end_tuple(std::index_sequence<Is...>) {
    return std::tuple(std::end(std::get<Is>(containers))...);
  }

public:
  explicit zip(Containers&... containers) : containers(containers...) {}

  class iterator {
  private:
    std::tuple<decltype(std::begin(std::declval<Containers&>()))...> iterators;

  public:
    explicit iterator(decltype(iterators) iterators) : iterators(std::move(iterators)) {}

    iterator& operator++() {
      std::apply([](auto&... it) { ((++it), ...); }, iterators);
      return *this;
    }

    auto operator*() {
      return std::apply([](auto&... it) { return std::tie(*it...); }, iterators);
    }

    bool operator!=(const iterator& other) const {
      return iterators != other.iterators;
    }
  };

  auto begin() {
    return iterator(make_begin_tuple(std::index_sequence_for<Containers...>{}));
  }

  auto end() {
    return iterator(make_end_tuple(std::index_sequence_for<Containers...>{}));
  }
};

template <typename T>
struct std::formatter<std::vector<T>> {
  char opening = '[';
  char closing = ']';
  std::string separator = ", ";

  constexpr auto parse(std::format_parse_context &ctx) { return ctx.begin(); }

  template <typename FormatContext>
  auto format(const std::vector<T> &vec, FormatContext &ctx) const {
    auto out = ctx.out();
    out = std::format_to(out, "{}", opening);
    for (size_t i = 0; i < vec.size(); ++i) {
      if (i > 0) out = std::format_to(out, "{}", separator);
      out = std::format_to(out, "{}", vec[i]);
    }
    return std::format_to(out, "{}", closing);
  }
};

#endif
