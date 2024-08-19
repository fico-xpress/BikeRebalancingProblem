#include <xpress.hpp>
#include <stdexcept>
#include <unordered_map>
#include <chrono>
#include "DataFrame.h"

using namespace xpress;
using namespace xpress::objects;


std::vector<std::vector<double>> convertScenariosToMatrix(std::map<std::string, DataFrame> scenarios) {
    std::vector<std::vector<double>> matrix;

    for (auto& [name, df] : scenarios) {
        std::vector<double> classicNetColumn = df.getColumn<double>("CLASSIC_net");
        matrix.push_back(std::move(classicNetColumn));
    }

    return matrix;
}

void saveDoubleToInfoDf(DataFrame& infoDf, double value, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<double>{value});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_DEP_SingleU.csv";
        infoDf.toCsv(fileName.str());
    }
}


using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
void saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "'" << columnName << "' took " << duration << "ms (" << duration/1000.0 << "s)" << std::endl;

    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<long long>{duration});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_DEP_SingleU.csv";
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
        int nr_scenarios = 10;
        std::string stationDataFilename;
        std::vector<std::string> tripDataFilenames;

        if (nr_stations == -1) {
            // Station data file
            stationDataFilename = "./data/Station_Info.csv";
            // Generate trip data filenames counting down from 394
            for (int i = 394; i >= 394 - nr_scenarios/15; --i) {
                tripDataFilenames.push_back("./data/" + std::to_string(i) + "_Net_Data.csv");
            }
        } else {
            // Station data file with size
            stationDataFilename = "./data/Station_Info_size" + std::to_string(nr_stations) + ".csv";

            // Generate trip data filenames with size counting down from 394
            for (int i = 394; i >= 394 - nr_scenarios/15; --i) {
                tripDataFilenames.push_back("./data/" + std::to_string(i) + "_Net_Data_size" + std::to_string(nr_stations) + ".csv");
            }
        }

        // Station information data:
        DataFrame stationData = DataFrame::readCSV(stationDataFilename);
        stationData.convertColumnToDouble("nbDocks");
        std::vector<double> k_i2 = stationData.getColumn<double>("nbDocks");

        // Trip information data:
        std::vector<std::vector<double>> d_s_i2;
        for (const auto& filename : tripDataFilenames) {
            DataFrame tripData = DataFrame::readCSV(filename);
            tripData.convertColumnToDouble("CLASSIC_net");
            std::map<std::string, DataFrame> scenarios = tripData.groupBy<std::string>("date");
            std::vector<std::vector<double>> d_s_i_single = convertScenariosToMatrix(scenarios);
            d_s_i2.insert(d_s_i2.end(), d_s_i_single.begin(), d_s_i_single.end());
            if (d_s_i2.size() >= nr_scenarios) {
                break;
            }
        }


        std::vector<double> k_i = k_i2;
        std::vector<std::vector<double>> d_s_i;
        for (int i=0; i<nr_scenarios; i++) {
            d_s_i.push_back(d_s_i2[i]);
        }
        // std::vector<std::vector<double>> d_s_i = {d_s_i2[0]};
        // std::vector<std::vector<double>> d_s_i = {d_s_i2[d_s_i2.size()-1]};
        // std::vector<std::vector<double>> d_s_i = {d_s_i2[0], d_s_i2[1], d_s_i2[2], d_s_i2[3]};


        /******************  Data Initialization ******************************/
        // Right-hand coefficients b for each 1st-stage constraint j
        // std::vector<double> k_i     = { 10, 15,  20, 30 };
        // Right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        // std::vector<std::vector<double>> d_s_i   = {{ 4,  -4,  9,  -4 }};//, { 6,  -4,  10,  -7 }};

        int NR_STATIONS = k_i.size();
        int NR_SCENARIOS = d_s_i.size();
        int NR_BIKES = mySum(k_i) / 3 * 2;
        std::cout << "Nr scenarios: " << d_s_i.size() << std::endl;
        std::cout << "Nr stations: " << d_s_i[0].size() << std::endl;
        std::cout << "Nr bikes: " << NR_BIKES << std::endl;

        // Objective coefficients c for each first-stage decision variable x_i
        std::vector<double> c_i(NR_STATIONS, 10);
        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        // Objective coefficients for each second-stage variable u_i
        std::vector<double> q_i_1(NR_STATIONS, 10);
        // Objective coefficients for each second-stage variable o_i
        std::vector<double> q_i_2(NR_STATIONS, 10);
        // Probability of each scenario s
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));

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
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int i){ return k_i[i]; })
            .withName([](int i){ return xpress::format("x_%d", i); })
            .toArray();

        // Create recourse variables y
        std::vector<std::vector<std::vector<Variable>>> y = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int s, int i, int j){ return k_i[i]; })
            // .withName([](int s, int i, int j){ return xpress::format("s%d_y_(%d,%d)", s, i, j); })
            .toArray();

        // Create station overflow helper variables o
        std::vector<std::vector<std::vector<Variable>>> o = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB([](int s, int i, int j) { return j%2 == 0 ? 0.0 : XPRS_MINUSINFINITY; })
            .withUB([&](int s, int i, int j){ return k_i[i] + std::abs(d_s_i[s][i]); })
            // .withName([](int s, int i, int j){ return xpress::format("s%d_o%s_%d", s, j%2==0 ? "Pos" : "Neg", i); })
            .toArray();

        // Create unmet demand helper variables u
        std::vector<std::vector<std::vector<Variable>>> u = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB([](int s, int i, int j) { return j%2 == 0 ? 0.0 : XPRS_MINUSINFINITY; })
            .withUB([&](int s, int i, int j){ return k_i[i] + std::abs(d_s_i[s][i]); })
            // .withName([](int s, int i, int j){ return xpress::format("s%d_u%s_%d", s, j%2==0 ? "Pos" : "Neg", i); })
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

        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                LinExpression net_recourse_flow = LinExpression::create();
                for (int j=0; j<NR_STATIONS; j++) {
                    net_recourse_flow.addTerm(y[s][i][j], 1).addTerm(y[s][j][i], -1);
                }
                end_of_day_net_recourse_flows[s][i] = net_recourse_flow;
                during_day_net_customer_flows[s][i] = -( d_s_i[s][i] - u[s][i][0] + o[s][i][0] );
            }
        }
        // prob.addConstraint(Utils::sum(during_day_net_customer_flows) == 0.0);
        // prob.addConstraint(Utils::sum(end_of_day_net_recourse_flows) == 0.0);

        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (end_of_day_net_recourse_flows[s][i] == during_day_net_customer_flows[s][i]);
                    // .setName(xpress::format("s%d_FlowCons_S%d", s, i));
        });

        // Indicators with sos1
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) { 
        //     return SOS::sos1(u[s][i], std::vector<double>{0.0, 1.0}, xpress::format("s%d_sos1_u%d", s, i)); });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) { 
        //     return SOS::sos1(o[s][i], std::vector<double>{0.0, 1.0}, xpress::format("s%d_sos1_o%d", s, i)); });
        // Indicators with bigM
        // std::vector<std::vector<xpress::objects::Variable>> indicators = prob.addVariables(2, NR_STATIONS).withType(ColumnType::Binary)
        //     .withName([](int i, int j){ return xpress::format("%sBool_%d", i%2==0 ? "u" : "o", j); }) .toArray();
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[0][i].ifThen(u[i][1] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[0][i].ifNotThen(u[i][0] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[1][i].ifThen(o[i][1] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[1][i].ifNotThen(o[i][0] == 0.0); });

        // u+[i] - u-[i] == d_i_s[i] - x[i]
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (u[s][i][1] == d_s_i[s][i] - x[i]);
        });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        //     return (u[s][i][0].maxOf(std::vector<Variable>{u[s][i][1]}, 0.0));
        // });
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (u[s][i][0] >= u[s][i][1]); });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        //     return (u[s][i][0] >= 0.0); });

        // o+[i] - o-[i] == -d_i_s[i] - (k[i] - x[i])
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (o[s][i][1] == -d_s_i[s][i] - (k_i[i] - x[i]));
        });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        //     return (o[s][i][0].maxOf(std::vector<Variable>{o[s][i][1]}, 0.0));
        // });
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (o[s][i][0] >= o[s][i][1]); });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
        //     return (o[s][i][0] >= 0.0); });

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
                for (int j=0; j<NR_STATIONS; j++) {
                    obj.addTerm(p_s[s] * c_ij[i][j], y[s][i][j]);
                }
                obj.addTerm(p_s[s] * q_i_1[i], u[s][i][0]);
                obj.addTerm(p_s[s] * q_i_2[i], o[s][i][0]);
            }
            // obj.addTerms(Utils::scalarProduct(q_i_1[i], u[s]), p_s[s]);
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


