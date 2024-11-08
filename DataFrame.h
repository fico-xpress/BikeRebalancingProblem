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
    // The only accepted data types by this DataFrame class
    using ColDataType = std::variant<
                            std::vector<std::string>,
                            std::vector<double>,
                            std::vector<long long>,
                            std::vector<int>
                        >;

    // Functions to add an empty column or add a column with provided values
    template<typename T>
    void addColumn(const std::string& name);
    void addColumn(const std::string& columnName, const ColDataType& columnValues);

    // Functions to get a column by name and check if a column with the given name exists
    template<typename T>
    std::vector<T> getColumn(const std::string& columnName) const;
    ColDataType getColumn(const std::string& columnName) const;
    bool hasColumnName(const std::string& name) const;

    // Function to convert a column of type std::string to a column of type double
    void convertStringColumnToDouble(const std::string& columnName);

    // Function to group the data by the key column
    template<typename T>
    std::map<T, DataFrame> groupBy(const std::string& keyColumnName) const;

    // Convert a CSV file to a DataFrame object or other way around
    static DataFrame readCSV(const std::string& filename);
    void toCsv(const std::string& filename) const;
    void toCsv(const std::string& filename, const char delim) const;

    // Some utility functions
    size_t length() const;
    void printColumnSizes() const;
    std::vector<std::string> columnNames() const;

private:
    std::map<std::string, ColDataType> columns;

    // Helper function to group the data by the key column
    template<typename KeyType, typename GroupType>
    void groupByHelper(const std::string colName, const ColDataType& colValues, 
                       const std::vector<KeyType>& keyColumn, std::map<KeyType, DataFrame>& groupedData) const;
};


// Namespace to hold utility functions for ColDataType
namespace ColDataTypeUtils {
    // Function to get the size of the column
    size_t size(const DataFrame::ColDataType& columnVariant) {
        return std::visit([](const auto& col) { return col.size(); }, columnVariant);
    }

    // Function to add an element to the column
    template<typename T>
    void addElement(DataFrame::ColDataType& data, const T& element) {
        if (std::holds_alternative<std::vector<T>>(data)) {
            std::get<std::vector<T>>(data).push_back(element);
        } else {
            throw std::invalid_argument("Element type does not match column type");
        }
    }
}


// Function to add an empty column to the DataFrame
template<typename T>
void DataFrame::addColumn(const std::string& name) {
    columns[name] = std::vector<T>{};
}

// Function to add a column with values to the DataFrame
void DataFrame::addColumn(const std::string& name, const ColDataType& columnValues) {
    columns[name] = columnValues;
}

// Function to get the number of rows in the DataFrame
size_t DataFrame::length() const {
    if (columns.empty()) return 0;
    return ColDataTypeUtils::size(columns.begin()->second);
}

// Function to print the sizes of all columns in the DataFrame
void DataFrame::printColumnSizes() const {
    for (const auto& col : columns) {
        std::cout << "\t" << col.first << ": " << ColDataTypeUtils::size(col.second) << std::endl;
    }
}

// Function to get the names of all columns in the DataFrame
std::vector<std::string> DataFrame::columnNames() const {
    std::vector<std::string> keys;
    for (const auto& col : columns) {
        keys.push_back(col.first);
    }
    return keys;
}

// Function to check if a column with the given name exists in the DataFrame
bool DataFrame::hasColumnName(const std::string& name) const {
    auto it = columns.find(name);
    if (it != columns.end()) {
        return true;
    }
    return false;
}

// Function to get the column with the given name. Return as std::variant type
DataFrame::ColDataType DataFrame::getColumn(const std::string& name) const {
    auto it = columns.find(name);
    if (it != columns.end()) {
        return it->second;
    }
    throw std::runtime_error("Column '" + name + "' not found");
}

// Function to get the column with the given name. Return as std::vector<T> type
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

// Function to read a CSV file and return a DataFrame object. All columns will be of type std::string
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

// Convenience function to write the DataFrame to a CSV file with "," as delimiter
void DataFrame::toCsv(const std::string& filename) const {
    toCsv(filename, ',');
}

// Function to write the DataFrame to a CSV file with 'delim' as delimiter
void DataFrame::toCsv(const std::string& filename, const char delim) const {
    std::ofstream file(filename);
    if (!file.is_open()) {
        throw std::runtime_error("Could not open file for writing");
    }

    // Write headers
    bool firstColumn = true;
    for (const auto& col : columns) {
        if (!firstColumn) {
            file << delim;
        }
        file << col.first;
        firstColumn = false;
    }
    file << "\n";

    // Determine the number of rows
    size_t numRows = length();

    // Write rows
    for (size_t i = 0; i < numRows; i++) {
        bool firstCell = true;
        for (const auto& col : columns) {
            if (!firstCell) {
                file << delim;
            }
            std::visit([i, &file](const auto& colVec) {
                file << colVec[i];
            }, col.second);
            firstCell = false;
        }
        file << "\n";
    }

    file.close();
}

// Function to convert a string column to a double column
void DataFrame::convertStringColumnToDouble(const std::string& columnName) {
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


// Helper function to group the data by the key column. The column type of keys to group by can be
// any from the variant, the same the type of values in the column to group by
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

// Function to group the data by the key column. The type of this key column must match the template type T,
// but can be any of the types in the variant ColDataType
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

    // Step 4: Loop over every column and group the data by the key column
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
