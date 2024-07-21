#ifndef DATAFRAME_H
#define DATAFRAME_H

#include <vector>
#include <string>
#include <unordered_map>
#include <map>
#include <variant>
#include <fstream>
#include <sstream>
#include <iostream>
#include <set>

class DataFrame {
public:
    using ColDataType = std::variant<std::vector<std::string>, std::vector<double>>;

    template<typename T>
    void addColumn(const std::string& name);
    void addColumn(const std::string& columnName, const ColDataType& column);

    template<typename T>
    std::vector<T> getColumn(const std::string& columnName) const;
    ColDataType getColumn(const std::string& columnName) const;

    void convertColumnToDouble(const std::string& columnName);

    template<typename T>
    std::map<T, DataFrame> groupBy(const std::string& columnName) const;

    static DataFrame readCSV(const std::string& filename);

    int length();
    void printColumnSizes();
    std::vector<std::string> columnNames();

private:
    std::map<std::string, ColDataType> columns;

    template<typename KeyType, typename GroupType>
    void groupByHelper(const std::string colName, const ColDataType& colValues, const std::vector<KeyType>& keyColumn, std::map<KeyType, DataFrame>& groupedData) const;

};











// Namespace to hold utility functions for ColDataType
namespace ColDataTypeUtils {
    // Function to get the size of the column
    size_t size(const DataFrame::ColDataType& columnVariant) {
        return std::visit([](const auto& col) { return col.size(); }, columnVariant);
    }

    // // Function to add an element to the column
    // template<typename T>
    // void addElement(DataFrame::ColDataType& data, const T& element) {
    //     if (std::holds_alternative<std::vector<T>>(data)) {
    //         std::get<std::vector<T>>(data).push_back(element);
    //     } else {
    //         throw std::invalid_argument("Element type does not match column type");
    //     }
    // }

    // // Function to remove an element from the column by index
    // void removeElement(DataFrame::ColDataType& data, size_t index) {
    //     std::visit([index](auto& col) {
    //         if (index < col.size()) {
    //             col.erase(col.begin() + index);
    //         } else {
    //             throw std::out_of_range("Index out of range");
    //         }
    //     }, data);
    // }

    // // Function to print the column data
    // void print(const DataFrame::ColDataType& data) {
    //     std::visit([](const auto& col) {
    //         for (const auto& elem : col) {
    //             std::cout << elem << " ";
    //         }
    //         std::cout << std::endl;
    //     }, data);
    // }
}









template<typename T>
void DataFrame::addColumn(const std::string& name) {
    columns[name] = std::vector<T>{};
}


void DataFrame::addColumn(const std::string& name, const ColDataType& column) {
    columns[name] = column;
}


int DataFrame::length() {
    if (columns.empty()) return 0;
    return ColDataTypeUtils::size(columns.begin()->second);
}

void DataFrame::printColumnSizes() {
    for (const auto& col : columns) {
        std::cout << "\t" << col.first << ": " << ColDataTypeUtils::size(col.second) << std::endl;
    }
}

std::vector<std::string> DataFrame::columnNames() {
    std::vector<std::string> keys;
    for (const auto& col : columns) {
        keys.push_back(col.first);
    }
    return keys;
}


DataFrame::ColDataType DataFrame::getColumn(const std::string& name) const {
    auto it = columns.find(name);
    if (it != columns.end()) {
        return it->second;
    }
    throw std::runtime_error("Column '" + name + "' not found");
}


template<typename T>
std::vector<T> DataFrame::getColumn(const std::string& name) const {
    auto it = columns.find(name);
    if (it != columns.end()) {
        if (!std::holds_alternative<std::vector<T>>(it->second)) {
            throw std::invalid_argument("Column type does not match template type T");
        }
        return std::get<std::vector<T>>(it->second);
    }
    throw std::runtime_error("Column '" + name + "' not found");
}


DataFrame DataFrame::readCSV(const std::string &filename) {
    DataFrame df;
    std::ifstream file(filename);

    if (!file.is_open()) {
        std::cerr << "Could not open the file: " << filename << std::endl;
        return df;
    }

    std::string line;
    bool isHeader = true;
    std::vector<std::string> headers;

    while (std::getline(file, line)) {
        std::stringstream lineStream(line);
        std::string cell;
        std::vector<std::string> row;

        while (std::getline(lineStream, cell, ';')) {
            row.push_back(cell);
        }

        if (isHeader) {
            headers = row;
            for (const auto& header : headers) {
                df.addColumn<std::string>(header);//, std::vector<std::string>{});
            }
            isHeader = false;
        } else {
            for (size_t i = 0; i < row.size(); i++) {
                std::get<std::vector<std::string>>(df.columns[headers[i]]).push_back(row[i]);
            }
        }
    }

    file.close();
    return df;
}


void DataFrame::convertColumnToDouble(const std::string& columnName) {
    auto columnValues = getColumn(columnName);

    if (std::holds_alternative<std::vector<double>>(columnValues)) {
        throw std::runtime_error("Column '" + columnName + "' is already of type <double>, no need to convert");
    }
    if (!std::holds_alternative<std::vector<std::string>>(columnValues)) {
        throw std::runtime_error("Column " + columnName + " is not of type <std::string>");
    }

    std::vector<std::string> stringColumn = std::get<std::vector<std::string>>(columnValues);
    std::vector<double> doubleColumn;
    doubleColumn.reserve(stringColumn.size());

    for (const std::string& str : stringColumn) {
        doubleColumn.push_back(std::stod(str));
    }

    // Replace the string column with the double column
    columns[columnName] = doubleColumn;
}



template<typename KeyType, typename GroupType>
void DataFrame::groupByHelper(const std::string colName, const ColDataType& colValues, const std::vector<KeyType>& keyColumn, std::map<KeyType, DataFrame>& groupedData) const {
    std::vector<GroupType> convertedColValues = std::get<std::vector<GroupType>>(colValues);
    std::unordered_map<KeyType, std::vector<GroupType>> colGroups;

    for (size_t i = 0; i < convertedColValues.size(); ++i) {
        colGroups[keyColumn[i]].push_back(convertedColValues[i]);
    }

    for (auto& [groupKey, values] : colGroups) {
        groupedData[groupKey].columns[colName] = std::move(colGroups[groupKey]);
    }
}


template<typename KeyType>
std::map<KeyType, DataFrame> DataFrame::groupBy(const std::string& keyColumnName) const {
    const ColDataType& keyColumnValues = getColumn(keyColumnName);

    if (!std::holds_alternative<std::vector<KeyType>>(keyColumnValues)) {
        throw std::invalid_argument("Column type does not match template type T");
    }
    std::vector<KeyType> keyColumn = std::get<std::vector<KeyType>>(keyColumnValues);

    // Step 1: Get unique values in the column. These will form the groups
    std::set<KeyType> uniqueKeys(keyColumn.begin(), keyColumn.end());

    // Step 2: Initialize an empty template DataFrame (to be used for each group)
    DataFrame templateDataFrame;
    for (const auto& [origColName, origColValues] : columns) {
        if (origColName == keyColumnName) continue;

        std::visit([&](const auto& colValuesVariant) {
            // Extract whether the variant contains vector<string> or vector<double>
            using ColType = std::decay_t<decltype(colValuesVariant)>::value_type;
            // Add new column with the correct data type
            templateDataFrame.addColumn<ColType>(origColName);
        }, origColValues);
    }

    // Step 3: Initialize DataFrames for each unique key
    std::map<KeyType, DataFrame> groupedData;
    for (const KeyType& key : uniqueKeys) {
        groupedData[key] = templateDataFrame;
    }

    // Step 4
    for (const auto& [colName, colValues] : columns) {
        if (colName == keyColumnName) continue;

        std::visit([&](const auto& colValuesVariant) {
            // Extract whether the variant contains vector<string> or vector<double>
            using GroupType = std::decay_t<decltype(colValuesVariant)>::value_type;
            // Add new column with the correct data type
            groupByHelper<KeyType,GroupType>(colName, colValues, keyColumn, groupedData);
        }, colValues);
    }

    return groupedData;
}





#endif // DATAFRAME_H
