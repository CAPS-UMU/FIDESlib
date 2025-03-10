#include "data_loading.hpp"

#include <fstream>
#include <vector>

bool load_csv(const std::string &filename, std::vector<std::vector<double> > &data) {

    data.clear();

    std::ifstream file(filename);
    if (!file.is_open()) {
        return false;
    }

    std::string line;
    while (std::getline(file, line)) {
        std::vector<double> row;
        std::string cell;
        for (char c: line) {
            if (c == ',') {
                row.push_back(stod(cell));
                cell.clear();
            } else {
                cell += c;
            }
        }
        row.push_back(stod(cell));
		data.push_back(row);
    }

    return true;
}
