#ifndef CSVLOADER_H
#define CSVLOADER_H

#include <fstream>
#include <iostream>
#include <sstream>
#include <stdexcept>
#include <string>
#include <vector>

struct SeriesData {
    std::vector<double> y;
    std::vector<double> x;
};

SeriesData load_series_csv(const std::string &filename) {
    SeriesData data;
    std::ifstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return data;
    }
    std::string line;
    while (std::getline(file, line)) {
        if (line.empty()) continue;
        std::stringstream ss(line);
        std::string cell;
        std::vector<double> row;
        while (std::getline(ss, cell, ',')) {
            try {
                row.push_back(std::stod(cell));
            } catch (...) {
            }
        }
        if (!row.empty()) data.y.push_back(row[0]);
        if (row.size() > 1) data.x.push_back(row[1]);
        else data.x.push_back(0.0);
    }
    return data;
}


#endif
