#include <xpress.hpp>
#include <stdexcept>
#include <unordered_map>
#include <chrono>
#include "DataFrame.h"

using namespace xpress;
using namespace xpress::objects;


using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
void saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "'" << columnName << "' took " << duration << "ms (" << duration/1000.0 << "s)" << std::endl;

    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<long long>{duration});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_DEP_DoubleU.csv";
        infoDf.toCsv(fileName.str());
    }
}

void saveDoubleToInfoDf(DataFrame& infoDf, double value, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<double>{value});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_DEP_DoubleU.csv";
        infoDf.toCsv(fileName.str());
    }
}

double mySum(std::vector<double> a) {
    double ans = 0.0;
    for (double val : a) ans += val;
    return ans;
}


int main() {

    bool SOLVE_LP_RELAXATION = false;

    try {
        // For keeping track of timings and other info
        DataFrame infoDf;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        // Count duration of data initialization
        start = std::chrono::high_resolution_clock::now();

        int nr_stations = -1;
        int nr_scenarios = 50;
        std::string tripDataFilename, stationDataFilename;
        if (nr_stations == -1) {
            stationDataFilename = "./data/Station_Info.csv";
        } else {
            stationDataFilename = "./data/Station_Info_size" + std::to_string(nr_stations) + ".csv";
        }

        // Station information data:
        DataFrame stationData = DataFrame::readCSV(stationDataFilename);
        stationData.convertColumnToDouble("nbDocks");
        std::vector<double> k_i2 = stationData.getColumn<double>("nbDocks");

        // Trip information data:
        std::vector<DataFrame> scenarioData;
        for (int day=1; day<nr_scenarios+1; day++) {
            if (nr_stations == -1) {
                tripDataFilename = "./data/matrix_data/matrix_data_" + std::to_string(day) + ".csv";
            } else {
                tripDataFilename = "./data/matrix_data/matrix_data_size" + std::to_string(nr_stations) + "_" + std::to_string(day) + ".csv";
            }
            DataFrame tripData = DataFrame::readCSV(tripDataFilename);
            for (std::string colName : tripData.columnNames()) {
                tripData.convertColumnToDouble(colName);
            }
            scenarioData.push_back(tripData);
        }

        // Convert column-wise dataframe to row-wise matrix
        int NR_SCENARIOS2 = scenarioData.size();
        int NR_STATIONS2 = scenarioData[0].length();
        std::vector<std::vector<std::vector<double>>> d_s_ij2(NR_SCENARIOS2, std::vector<std::vector<double>>(NR_STATIONS2, std::vector<double>(NR_STATIONS2)));
        for (int s=0; s<NR_SCENARIOS2; s++) {
            for (std::string colName : scenarioData[s].columnNames()) {
                int colIndex = std::stoi(colName);
                std::vector<double> colValues = scenarioData[s].getColumn<double>(colName);
                for (int i=0; i<NR_STATIONS2; i++) {
                    d_s_ij2[s][i][colIndex] = colValues[i];
                }
            }
        }

        std::vector<double> k_i = k_i2;
        std::vector<std::vector<std::vector<double>>> d_s_ij;
        for (int i=0; i<nr_scenarios; i++) {
            d_s_ij.push_back(d_s_ij2[i]);
        }
        // std::vector<std::vector<std::vector<double>>> d_s_ij = {d_s_ij2[0]};
        // std::vector<std::vector<std::vector<double>>> d_s_ij = {d_s_ij2[d_s_ij2.size()-1]};
        // std::vector<std::vector<std::vector<double>>> d_s_ij = {d_s_ij2[0], d_s_ij2[1], d_s_ij2[2], d_s_ij2[3]};



        /******************  Data Initialization ******************************/
        // Right-hand coefficients b for each 1st-stage constraint j
        // std::vector<double> k_i     = { 10, 15,  20, 30 };
        // Right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        // std::vector<std::vector<double>> d_s_i   = {{ 4,  -4,  9,  -4 }};//, { 6,  -4,  10,  -7 }};

        int NR_STATIONS = k_i.size();
        int NR_SCENARIOS = d_s_ij.size();
        int NR_BIKES = mySum(k_i) / 3 * 2;
        std::cout << "Nr scenarios: " << NR_SCENARIOS << std::endl;
        std::cout << "Nr stations: " << NR_STATIONS << std::endl;
        std::cout << "Nr bikes: " << NR_BIKES << std::endl;

        // Objective coefficients c for each first-stage decision variable x_i
        std::vector<double> c_i(NR_STATIONS, 10);
        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        // Objective coefficients for each second-stage variable u_i
        std::vector<std::vector<double>> q_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        // Probability of each scenario s
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));

        // Calculate net demands
        std::vector<std::vector<double>> netDemand_s_i;
        netDemand_s_i = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_STATIONS, 0));
        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                for (int j=0; j<NR_STATIONS; j++) {
                    netDemand_s_i[s][i] += d_s_ij[s][i][j];
                    netDemand_s_i[s][j] -= d_s_ij[s][i][j]; 
                }
            }
        }
        // End of data initialization
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Data Initialization (ms)", NR_STATIONS, NR_SCENARIOS);


        /******************  Problem Creation ******************************/
        // Create a problem instance
        XpressProblem prob;
        prob.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);


        std::cout << "CREATING VARIABLES" << std::endl;

        // Count duration of variable creation
        start = std::chrono::high_resolution_clock::now();

        /* VARIABLES */
        // Create first-stage variables x
        std::vector<Variable> x = prob.addVariables(NR_STATIONS)
            .withType(ColumnType::Integer)
            .withLB(0)
            .withUB([&](int i){ return k_i[i]; })
            .withName([](int i){ return xpress::format("x_%d", i); })
            .toArray();

        // Create recourse variables y
        std::vector<std::vector<std::vector<Variable>>> y = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Integer)
            .withLB(0)
            .withUB([&](int s, int i, int j){ return k_i[i]; })
            // .withName([](int s, int i, int j){ return xpress::format("s%d_y_(%d,%d)", s, i, j); })
            .toArray();

        // Create unmet demand helper variables u
        std::vector<std::vector<std::vector<Variable>>> u = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Integer)
            .withLB(0)
            // .withName([](int s, int i, int j){ return xpress::format("s%d_u_(%d,%d)", s, i, j); })
            .toArray();

        // End of variable creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Variable Creation (ms)", NR_STATIONS, NR_SCENARIOS);

        /* CONSTRAINTS */
        std::cout << "CREATING CONSTRAINTS" << std::endl;

        // Count duration of constraint creation
        start = std::chrono::high_resolution_clock::now();

        // First Stage constraints
        prob.addConstraint(Utils::sum(x) == NR_BIKES);
        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (x[i] <= k_i[i]).setName(xpress::format("Capacity%d", i));
        });

        // Second Stage constraints
        std::vector<std::vector<LinExpression>> end_of_day_net_recourse_flows(NR_SCENARIOS, std::vector<LinExpression>(NR_STATIONS));
        std::vector<std::vector<Expression>> during_day_net_customer_flows(NR_SCENARIOS, std::vector<Expression>(NR_STATIONS));
        std::vector<std::vector<LinExpression>> nr_disabled_incoming_trips(NR_SCENARIOS, std::vector<LinExpression>(NR_STATIONS));
        std::vector<std::vector<LinExpression>> nr_disabled_outgoing_trips(NR_SCENARIOS, std::vector<LinExpression>(NR_STATIONS));

        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                LinExpression net_recourse_flow = LinExpression::create();
                LinExpression outgoing_trips = LinExpression::create();
                LinExpression incoming_trips = LinExpression::create();
                for (int j=0; j<NR_STATIONS; j++) {
                    net_recourse_flow.addTerm(y[s][i][j], 1).addTerm(y[s][j][i], -1);
                    outgoing_trips.addTerm(u[s][i][j], 1);
                    incoming_trips.addTerm(u[s][j][i], 1);
                }
                end_of_day_net_recourse_flows[s][i] = net_recourse_flow;
                nr_disabled_outgoing_trips[s][i] = outgoing_trips;
                nr_disabled_incoming_trips[s][i] = incoming_trips;
                during_day_net_customer_flows[s][i] = -(netDemand_s_i[s][i] - nr_disabled_outgoing_trips[s][i] + nr_disabled_incoming_trips[s][i]);
            }
        }
        // prob.addConstraint(Utils::sum(during_day_net_customer_flows) == 0.0);
        // prob.addConstraint(Utils::sum(end_of_day_net_recourse_flows) == 0.0);

        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (end_of_day_net_recourse_flows[s][i] == during_day_net_customer_flows[s][i]);
                    // .setName(xpress::format("s%d_FlowCons_S%d", s, i));
        });
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (x[i] + during_day_net_customer_flows[s][i] <= k_i[i]);
        });
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (x[i] + during_day_net_customer_flows[s][i] >= 0);
        });

        // End of constraint creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Constraint Creation (ms)", NR_STATIONS, NR_SCENARIOS);


        /* OBJECTIVE */
        std::cout << "CREATING OBJECTIVE" << std::endl;

        // Count duration of objective creation
        start = std::chrono::high_resolution_clock::now();

        LinExpression obj = LinExpression::create();
        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                // obj.addTerms(Utils::scalarProduct(c_ij[i], y[s][i]), p_s[s]);
                // obj.addTerms(Utils::scalarProduct(q_ij[i], u[s][i]), p_s[s]);
                for (int j=0; j<NR_STATIONS; j++) {
                    obj.addTerm(p_s[s] * c_ij[i][j], y[s][i][j]);
                    obj.addTerm(p_s[s] * q_ij[i][j], u[s][i][j]);
                }
            }
        }
        LinExpression firstStageCosts = Utils::scalarProduct(x, c_i);
        obj.addTerms(firstStageCosts);

        prob.setObjective(obj, xpress::ObjSense::Minimize);

        // End of objective creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Objective Creation (ms)", NR_STATIONS, NR_SCENARIOS);

        /* INSPECT, SOLVE & PRINT */

        // Solve the problem
        std::cout << "Solving the problem" << std::endl;
        // Count duration of Optimization
        start = std::chrono::high_resolution_clock::now();

        // Optimize
        if (SOLVE_LP_RELAXATION) prob.lpOptimize();
        else prob.optimize();

        // End of Optimization
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Optimization (ms)", NR_STATIONS, NR_SCENARIOS);
        saveDoubleToInfoDf(infoDf, prob.getObjVal(), "ObjectiveVal", NR_STATIONS, NR_SCENARIOS);

        // Check the solution status
        if (prob.getSolStatus() != SolStatus::Optimal && prob.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << prob.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Print the solution to console (first set precision to e.g. 5)
        std::cout << std::endl << "*** Objective Value ***" << std::endl;
        std::cout << "Solution has objective value (costs) of " << prob.getObjVal() << std::endl;
        std::cout << std::endl << "*** Solution of the " << (SOLVE_LP_RELAXATION ? "LP RELAXATION" : "ORIGINAL problem");
        std::cout << " ***" << std::endl;

        // Retrieve the solution values in one go
        std::vector<double> sol = prob.getSolution();

        // Loop over the relevant variables and print their name and value
        for (Variable x_i : x) std::cout << x_i.getName() << " = " << x_i.getValue(sol) << std::endl;
        std::cout << std::endl;

        std::cout << "2nd Stage Costs = " << prob.getObjVal() - firstStageCosts.evaluate(sol) << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}


