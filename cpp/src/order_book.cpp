// Price-time-priority limit order book, C++ port of quantsim/exchange/order_book.py.
//
// This exists purely as an optional systems-engineering exercise: the
// pure-Python OrderBook already replays comfortably fast for this project's
// data volumes, so this module is not a dependency of anything else in
// quantsim -- see scripts/demo_cpp_benchmark.py for the throughput comparison
// this was actually built to demonstrate.
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include <algorithm>
#include <cstdint>
#include <deque>
#include <map>
#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace py = pybind11;

struct RestingOrder {
    int64_t order_id;
    std::string side;
    double price;
    double quantity;
    py::object timestamp;
};

struct Trade {
    int64_t resting_order_id;
    double price;
    double quantity;
    py::object timestamp;
};

class OrderBook {
public:
    OrderBook() : next_id_(1) {}

    std::optional<double> best_bid() const {
        if (bids_.empty()) return std::nullopt;
        return bids_.rbegin()->first;
    }

    std::optional<double> best_ask() const {
        if (asks_.empty()) return std::nullopt;
        return asks_.begin()->first;
    }

    std::pair<int64_t, std::vector<Trade>> add_limit(
        const std::string& side, double price, double quantity, py::object timestamp,
        std::optional<int64_t> order_id_opt) {
        int64_t order_id = order_id_opt.value_or(next_id_++);
        auto [trades, remaining] = match(side, quantity, timestamp, price);

        if (remaining > 0) {
            auto& book = (side == "BUY") ? bids_ : asks_;
            book[price].push_back({order_id, side, price, remaining, timestamp});
            order_index_[order_id] = {side, price};
        }
        return {order_id, trades};
    }

    bool cancel(int64_t order_id) {
        auto it = order_index_.find(order_id);
        if (it == order_index_.end()) return false;

        auto [side, price] = it->second;
        order_index_.erase(it);
        auto& book = (side == "BUY") ? bids_ : asks_;
        auto level_it = book.find(price);
        if (level_it == book.end()) return false;

        auto& level = level_it->second;
        for (auto order_it = level.begin(); order_it != level.end(); ++order_it) {
            if (order_it->order_id == order_id) {
                level.erase(order_it);
                break;
            }
        }
        if (level.empty()) book.erase(level_it);
        return true;
    }

    std::pair<std::vector<Trade>, double> market_order(const std::string& side, double quantity,
                                                         py::object timestamp) {
        return match(side, quantity, timestamp, std::nullopt);
    }

    std::pair<std::vector<Trade>, double> ioc_order(const std::string& side, double quantity, double price,
                                                      py::object timestamp) {
        return match(side, quantity, timestamp, price);
    }

    // Runs a whole batch of (is_market, side, price, quantity) operations
    // without crossing back into Python between operations. A benchmark that
    // calls market_order/add_limit once per Python-loop iteration measures
    // pybind11 call overhead as much as the matching logic itself; this is
    // the fair way to measure the matching loop's own throughput. See
    // scripts/demo_cpp_benchmark.py. Uses a shared dummy timestamp since the
    // benchmark only cares about matching throughput, not timestamps.
    int64_t run_batch(const std::vector<std::tuple<bool, std::string, double, double>>& ops) {
        py::object dummy_timestamp = py::none();
        int64_t total_trades = 0;

        for (const auto& [is_market, side, price, quantity] : ops) {
            std::optional<double> price_limit = is_market ? std::nullopt : std::optional<double>(price);
            auto [trades, remaining] = match(side, quantity, dummy_timestamp, price_limit);
            total_trades += static_cast<int64_t>(trades.size());

            if (!is_market && remaining > 0) {
                int64_t order_id = next_id_++;
                auto& book = (side == "BUY") ? bids_ : asks_;
                book[price].push_back({order_id, side, price, remaining, dummy_timestamp});
                order_index_[order_id] = {side, price};
            }
        }
        return total_trades;
    }

private:
    using Level = std::deque<RestingOrder>;
    using Book = std::map<double, Level>;

    std::pair<std::vector<Trade>, double> match(const std::string& side, double quantity, py::object timestamp,
                                                  std::optional<double> price_limit) {
        std::vector<Trade> trades;
        double remaining = quantity;
        bool is_buy = (side == "BUY");
        Book& book = is_buy ? asks_ : bids_;

        if (is_buy) {
            for (auto it = book.begin(); it != book.end() && remaining > 0;) {
                double price = it->first;
                if (price_limit && price > *price_limit) break;
                remaining = consume_level(it->second, remaining, trades, timestamp, price);
                it = it->second.empty() ? book.erase(it) : std::next(it);
            }
        } else {
            for (auto it = book.rbegin(); it != book.rend() && remaining > 0;) {
                double price = it->first;
                if (price_limit && price < *price_limit) break;
                remaining = consume_level(it->second, remaining, trades, timestamp, price);
                if (it->second.empty()) {
                    it = Book::reverse_iterator(book.erase(std::next(it).base()));
                } else {
                    ++it;
                }
            }
        }
        return {trades, remaining};
    }

    double consume_level(Level& level, double remaining, std::vector<Trade>& trades, py::object timestamp,
                          double price) {
        while (!level.empty() && remaining > 0) {
            RestingOrder& resting = level.front();
            double fill_qty = std::min(resting.quantity, remaining);
            trades.push_back({resting.order_id, price, fill_qty, timestamp});
            resting.quantity -= fill_qty;
            remaining -= fill_qty;
            if (resting.quantity <= 0) {
                order_index_.erase(resting.order_id);
                level.pop_front();
            }
        }
        return remaining;
    }

    Book bids_;
    Book asks_;
    std::unordered_map<int64_t, std::pair<std::string, double>> order_index_;
    int64_t next_id_;
};

PYBIND11_MODULE(quantsim_matching_engine, m) {
    m.doc() = "C++ price-time-priority order book (optional accelerated backend for quantsim.exchange.order_book)";

    py::class_<Trade>(m, "Trade")
        .def_readonly("resting_order_id", &Trade::resting_order_id)
        .def_readonly("price", &Trade::price)
        .def_readonly("quantity", &Trade::quantity)
        .def_readonly("timestamp", &Trade::timestamp);

    py::class_<OrderBook>(m, "OrderBook")
        .def(py::init<>())
        .def("best_bid", &OrderBook::best_bid)
        .def("best_ask", &OrderBook::best_ask)
        .def("add_limit", &OrderBook::add_limit, py::arg("side"), py::arg("price"), py::arg("quantity"),
             py::arg("timestamp"), py::arg("order_id") = py::none())
        .def("cancel", &OrderBook::cancel, py::arg("order_id"))
        .def("market_order", &OrderBook::market_order, py::arg("side"), py::arg("quantity"), py::arg("timestamp"))
        .def("ioc_order", &OrderBook::ioc_order, py::arg("side"), py::arg("quantity"), py::arg("price"),
             py::arg("timestamp"))
        .def("run_batch", &OrderBook::run_batch, py::arg("operations"));
}
