#pragma once
/**
 * spin_table.h — the text format of spin configurations (initial_spin_config,
 * spins_*.txt, initial_spins*.txt), shared by Lattice, MixedLattice and
 * PhononLattice.
 *
 * Format: one site per line, `dim` whitespace-separated numbers; '#' starts a
 * comment, blank lines are skipped. Writers print max_digits10 significant
 * digits, so a saved configuration reloads bitwise.
 *
 * The reader is strict: a configuration that does not match the lattice
 * exactly is an error, never a partial overwrite (the old loaders printed to
 * stderr and returned, leaving random or half-loaded spins, and a run then
 * skipped equilibration because a file had been "loaded").
 */

#include <cmath>
#include <cstdlib>
#include <fstream>
#include <iomanip>
#include <limits>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

namespace classical_spin::io {

/// Rows of a numeric table and the file line each row came from.
struct Table {
    std::vector<double> values;   // n_rows * n_cols, row-major
    std::vector<size_t> lines;    // 1-based file line of each row
    size_t n_cols = 0;
    const double* row(size_t i) const { return values.data() + i * n_cols; }
};

/**
 * Read exactly n_rows rows of n_cols finite numbers. Throws
 * std::runtime_error naming the file (and line) for a missing file, a token
 * that is not a finite number, a row with the wrong number of columns, a
 * short file or extra rows.
 */
inline Table read_table(const std::string& filename, size_t n_rows, size_t n_cols) {
    std::ifstream in(filename);
    if (!in) throw std::runtime_error("cannot open spin configuration file '" + filename + "'");
    Table t;
    t.n_cols = n_cols;
    t.values.reserve(n_rows * n_cols);
    t.lines.reserve(n_rows);
    std::string line, tok;
    size_t line_no = 0;
    while (std::getline(in, line)) {
        ++line_no;
        const size_t hash = line.find('#');
        if (hash != std::string::npos) line.resize(hash);
        std::istringstream iss(line);
        size_t k = 0;
        const size_t start = t.values.size();
        while (iss >> tok) {
            char* end = nullptr;
            const double v = std::strtod(tok.c_str(), &end);
            if (end == tok.c_str() || *end != '\0' || !std::isfinite(v))
                throw std::runtime_error(filename + ":" + std::to_string(line_no) + ": not a finite number: '" +
                                         tok + "'");
            if (k == n_cols)
                throw std::runtime_error(filename + ":" + std::to_string(line_no) + ": more than " +
                                         std::to_string(n_cols) + " values (expected one spin of dimension " +
                                         std::to_string(n_cols) + " per line)");
            t.values.push_back(v);
            ++k;
        }
        if (k == 0) continue;   // blank or comment line
        if (k != n_cols)
            throw std::runtime_error(filename + ":" + std::to_string(line_no) + ": " + std::to_string(k) +
                                     " values, expected " + std::to_string(n_cols));
        if (t.lines.size() == n_rows) {
            t.values.resize(start);
            throw std::runtime_error(filename + ":" + std::to_string(line_no) + ": more than " +
                                     std::to_string(n_rows) + " spins (wrong lattice size?)");
        }
        t.lines.push_back(line_no);
    }
    if (t.lines.size() != n_rows)
        throw std::runtime_error(filename + ": " + std::to_string(t.lines.size()) + " spins, expected " +
                                 std::to_string(n_rows) + " (wrong lattice size or truncated file)");
    return t;
}

/**
 * Write n_rows lines of n_cols values value(i, j) at max_digits10 precision.
 * Throws std::runtime_error if the file cannot be written.
 */
template <class Value>
void write_table(const std::string& filename, size_t n_rows, size_t n_cols, Value&& value) {
    std::ofstream out(filename);
    if (!out) throw std::runtime_error("cannot open '" + filename + "' for writing");
    out << std::setprecision(std::numeric_limits<double>::max_digits10);
    for (size_t i = 0; i < n_rows; ++i) {
        for (size_t j = 0; j < n_cols; ++j) out << (j ? " " : "") << value(i, j);
        out << '\n';
    }
    out.close();
    if (!out) throw std::runtime_error("writing '" + filename + "' failed");
}

}  // namespace classical_spin::io
