#ifndef BRPUTILS_H
#define BRPUTILS_H

#include <vector>
#include <string>
#include <set>
#include <unordered_map>
#include <map>
#include <fstream>
#include <sstream>
#include <iostream>
#include <stdexcept> // For throwing exceptions
#include <chrono>    // For timekeeping
#include "DataFrame.h"

class BrpUtils {
public:
    // To save run information to a DataFrame
    using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
    static void saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName, std::string fileName);
    static void saveDoubleToInfoDf(DataFrame& infoDf, double value, std::string columnName, std::string fileName);

    // Some basic vector operations
    static double mySum(const std::vector<double> a);
    static std::vector<double> myElementWiseMultiplication(const double a, const std::vector<double>& b);
    static std::vector<double> myElementWiseMultiplication(const std::vector<double>& a, const std::vector<double>& b);
    static std::vector<double> myElementWiseAddition(const std::vector<double>& a, const std::vector<double>& b);
    static std::vector<double> myElementWiseSubtraction(const std::vector<double>& a, const std::vector<double>& b);

    static double myScalarProduct(const std::vector<double>& a, const std::vector<double>& b);
    static std::vector<std::vector<double>> myMatrixMultiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B);

    // Vector to output (CSV or console)
    static void writeVectorToCSV(std::vector<double>& vec, const std::string& filename);
    static void printDoubleVec(const std::vector<double> values, std::string delim);

    // To get data from files
    static std::vector<std::vector<std::vector<double>>> getTripsData(int nr_stations, int nr_scenarios);
    static std::vector<std::vector<double>> getNetTripsData(int nr_stations, int nr_scenarios);
    static std::vector<double> getStationInfoData(int nr_stations);
    static std::vector<std::vector<double>> getStationDistancesData(int nr_stations);
    
    static double getMaxDistance(const std::vector<std::vector<double>> c_ij);
    static std::vector<double> getAverageDistances(const std::vector<std::vector<double>> c_ij);
private:
};


using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
void BrpUtils::saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName, std::string fileName) {
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "\t'" << columnName << "' took " << duration << "ms (" << duration/1000.0 << "s)" << std::endl;

    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<long long>{duration});

        infoDf.toCsv("./data_out/" + fileName + ".csv");
    }
}

void BrpUtils::saveDoubleToInfoDf(DataFrame& infoDf, double value, std::string columnName, std::string fileName) {
    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<double>{value});

        infoDf.toCsv("./data_out/" + fileName + ".csv");
    }
}


double BrpUtils::mySum(std::vector<double> a) {
    double ans = 0.0;
    for (double val : a) ans += val;
    return ans;
}

std::vector<double> BrpUtils::myElementWiseMultiplication(const double a, const std::vector<double>& b) {
    std::vector<double> ans(b.size());
    for (int i=0 ; i<b.size(); i++) {
        ans[i] = a * b[i];
    }
    return ans;
}

std::vector<double> BrpUtils::myElementWiseMultiplication(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] * b[i];
    }
    return ans;
}

std::vector<double> BrpUtils::myElementWiseAddition(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}


std::vector<double> BrpUtils::myElementWiseSubtraction(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] - b[i];
    }
    return ans;
}

double BrpUtils::myScalarProduct(const std::vector<double>& a, const std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    double ans = 0.0;
    for (int i=0 ; i<a.size(); i++) {
        ans += a[i] * b[i];
    }
    return ans;
}

std::vector<std::vector<double>> BrpUtils::myMatrixMultiplication(const std::vector<std::vector<double>>& A, const std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) throw std::invalid_argument("Number of columns in A must be equal to the number of rows in B.");

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0));

    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

void BrpUtils::writeVectorToCSV(std::vector<double>& vec, const std::string& filename) {
    std::ofstream file(filename);

    if (file.is_open()) {
        for (size_t i = 0; i < vec.size(); ++i) {
            file << vec[i];
            if (i != vec.size() - 1) {
                file << ",";
            }
        }
        file << "\n";
        file.close();
    } else {
        std::cerr << "Unable to open file: " << filename << std::endl;
    }
}

void BrpUtils::printDoubleVec(const std::vector<double> values, std::string delim) {
    for (int i=0; i<values.size(); i++) {
        std::cout << values[i] << delim;
    }
    std::cout << std::endl;
}

std::vector<std::vector<double>> BrpUtils::getStationDistancesData(int nr_stations) {
    std::string distanceDataFilename;
    if (nr_stations == -1) {
        distanceDataFilename = "./data/Station_Distances.csv";
    } else {
        distanceDataFilename = "./data/Station_Distances_size" + std::to_string(nr_stations) + ".csv";
    }

    // Distances data:
    std::vector<std::vector<double>> c_ij;
    DataFrame distanceData = DataFrame::readCSV(distanceDataFilename, ';');
    for (std::string colName : distanceData.columnNames()) {
        distanceData.convertStringColumnToDouble(colName);
        std::vector<double> distVec = distanceData.getColumn<double>(colName);
        c_ij.push_back(std::move(distVec));
    }

    return c_ij;
}

std::vector<double> BrpUtils::getAverageDistances(const std::vector<std::vector<double>> c_ij) {
    std::vector<double> c_i(c_ij.size());
    for (int i=0; i<c_ij.size(); i++) {
        c_i[i] = BrpUtils::mySum(c_ij[i]) / c_ij.size() / c_ij.size();
    }
    return c_i;
}

double BrpUtils::getMaxDistance(const std::vector<std::vector<double>> c_ij) {
    double max_dist = 0.0;
    for (int i=0; i<c_ij.size(); i++) {
        double row_max = *std::max_element(c_ij[i].begin(), c_ij[i].end());
        max_dist = row_max > max_dist ? row_max : max_dist;
    }
    return max_dist;
}

std::vector<double> BrpUtils::getStationInfoData(int nr_stations) {
    std::string stationDataFilename;
    if (nr_stations == -1) {
        stationDataFilename = "./data/Station_Info.csv";
    } else {
        stationDataFilename = "./data/Station_Info_size" + std::to_string(nr_stations) + ".csv";
    }

    // Station information data:
    DataFrame stationData = DataFrame::readCSV(stationDataFilename, ';');
    stationData.convertStringColumnToDouble("nbDocks");
    std::vector<double> b_i = stationData.getColumn<double>("nbDocks");

    return b_i;
}

std::vector<std::vector<double>> BrpUtils::getNetTripsData(int nr_stations, int nr_scenarios) {
    std::vector<std::vector<std::vector<double>>> d_s_ij = BrpUtils::getTripsData(nr_stations, nr_scenarios);

    std::vector<std::vector<double>> d_s_i(nr_scenarios, std::vector<double>(nr_stations, 0));
    for (int s=0; s<nr_scenarios; s++) {
        for (int i=0; i<nr_stations; i++) {
            for (int j=0; j<nr_stations; j++) {
                d_s_i[s][i] += d_s_ij[s][i][j];
                d_s_i[s][j] -= d_s_ij[s][i][j]; 
            }
        }
    }
    return d_s_i;
}

std::vector<std::vector<std::vector<double>>> BrpUtils::getTripsData(int nr_stations, int nr_scenarios) {

    std::string tripDataFilename;

    // Trip information data:
    std::vector<DataFrame> scenarioData;
    for (int day=0; day<nr_scenarios; day++) {
        if (nr_stations == -1) {
            tripDataFilename = "./data/matrix_data/matrix_data_" + std::to_string(day) + ".csv";
        } else {
            tripDataFilename = "./data/matrix_data/matrix_data_size" + std::to_string(nr_stations) + "_" + std::to_string(day) + ".csv";
        }
        DataFrame tripData = DataFrame::readCSV(tripDataFilename, ';');
        for (int station_nr=0; station_nr<tripData.length(); station_nr++) {
            tripData.convertStringColumnToDouble(std::to_string(station_nr));
        }
        scenarioData.push_back(tripData);
    }

    // Convert column-wise dataframe to row-wise matrix
    int NR_SCENARIOS2 = scenarioData.size();
    int NR_STATIONS2 = scenarioData[0].length();
    std::vector<std::vector<std::vector<double>>> d_s_ij2(NR_SCENARIOS2, std::vector<std::vector<double>>(NR_STATIONS2, std::vector<double>(NR_STATIONS2)));
    for (int s=0; s<NR_SCENARIOS2; s++) {
        for (int station_nr=0; station_nr<NR_STATIONS2; station_nr++) {
            std::vector<double> colValues = scenarioData[s].getColumn<double>(std::to_string(station_nr));
            for (int i=0; i<NR_STATIONS2; i++) {
                d_s_ij2[s][i][station_nr] = colValues[i];
            }
        }
    }

    std::vector<std::vector<std::vector<double>>> d_s_ij(nr_scenarios);
    for (int s=0; s<nr_scenarios; s++) {
        d_s_ij[s] = d_s_ij2[s];
    }

    return d_s_ij;
}


#endif // BRPUTILS_H
