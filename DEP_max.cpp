#include <xpress.hpp>
#include <stdexcept>
#include <unordered_map>
#include <chrono>
#include "DataFrame.h" // Requires at least C++17

using namespace xpress;
using namespace xpress::objects;


// TODO: Investigate effect of using x*y == 0.0 instead of max(0.0, x) operator
// TODO: Investigate effect of using partial integer variables instead of continuous or integer variables
// TODO: Try making additional constraints for the LP-relaxation.

std::vector<std::vector<double>> convertScenariosToMatrix(std::map<std::string, DataFrame> scenarios) {
    std::vector<std::vector<double>> matrix;

    for (auto& [name, df] : scenarios) {
        // df.sortByColumn("station number");

        // std::vector<std::string> stationIDs = df.getColumn<std::string>("station number");
        // std::cout << stationIDs[0] << ", " << stationIDs[stationIDs.size()-1] << std::endl;

        std::vector<double> classicNetColumn = df.getColumn<double>("CLASSIC_net");

        matrix.push_back(std::move(classicNetColumn));
    }

    return matrix;
}

using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
void saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName) {
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "'" << columnName << "' took " << duration << "ms (" << duration/1000.0 << "s)" << std::endl;

    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<long long>{duration});
        infoDf.toCsv("./time_data/infoDf.csv");
    }
}



int main() {

    bool SOLVE_LP_RELAXATION = false;

    try {
        // For keeping track of timings and other info
        DataFrame infoDf;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;

        // Count duration of data initialization
        start = std::chrono::high_resolution_clock::now();

        std::string tripDataFilename = "./394_Net_Data.csv";
        std::string stationDataFilename = "./Station_Info.csv";
        DataFrame tripData    = DataFrame::readCSV(tripDataFilename);
        DataFrame stationData = DataFrame::readCSV(stationDataFilename);

        tripData.convertColumnToDouble("CLASSIC_net");
        stationData.convertColumnToDouble("nbDocks");

        std::map<std::string, DataFrame> scenarios = tripData.groupBy<std::string>("date");
        // for (auto& [scenarioName, scenarioValues] : scenarios) {
        //     std::cout << scenarioName << ": " << scenarioValues.length() << std::endl;
        //     scenarioValues.printColumnSizes();
        // }

        std::vector<double> k_i = stationData.getColumn<double>("nbDocks");
        std::vector<std::vector<double>> d_s_i2 = convertScenariosToMatrix(scenarios);
        std::vector<std::vector<double>> d_s_i = {d_s_i2[0]};//, d_s_i2[1], d_s_i2[2], d_s_i2[3]};

        std::cout << "Nr scenarios: " << d_s_i.size() << std::endl;
        std::cout << "Nr stations: " << d_s_i[0].size() << std::endl;

        // Dummy data:
        // std::vector<double> k_i     = { 10, 15,  20, 30 };
        // std::vector<std::vector<double>> d_s_i   = {{ 4,  -5,  9,  -8 }, { 6,  -4,  10,  -7 }};
        // int NR_BIKES = 46;

        int NR_STATIONS = k_i.size();
        int NR_SCENARIOS = d_s_i.size();

        std::vector<double> c_i(NR_STATIONS, 1);
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        std::vector<double> q_i_1(NR_STATIONS, 10);
        std::vector<double> q_i_2(NR_STATIONS, 10);

        // End of data initialization
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Data Initialization (ms)");


        // Create a problem instance
        XpressProblem prob;
        prob.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);


        std::cout << "CREATING VARIABLES" << std::endl;

        // Count duration of variable creation
        start = std::chrono::high_resolution_clock::now();

        /* VARIABLES */
        // Create first-stage variables x
        std::vector<xpress::objects::Variable> x = prob.addVariables(NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int i){ return k_i[i]; })
            .withName([](int i){ return xpress::format("x_%d", i); })
            .toArray();

        // end = std::chrono::high_resolution_clock::now();
        // saveTimeToInfoDf(infoDf, start, end, "Variables x (ms)");
        // start = std::chrono::high_resolution_clock::now();

        // Create recourse variables y
        std::vector<std::vector<std::vector<xpress::objects::Variable>>> y = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int s, int i, int j){ return k_i[i]; })
            .withName([](int s, int i, int j){ return xpress::format("y_%d_(%d,%d)", s, i, j); })
            .toArray();

        // end = std::chrono::high_resolution_clock::now();
        // saveTimeToInfoDf(infoDf, start, end, "Variables y (ms)");
        // start = std::chrono::high_resolution_clock::now();

        // Create station overflow helper variables o
        std::vector<std::vector<std::vector<xpress::objects::Variable>>> o = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB([](int s, int i, int j) { return j%2 == 0 ? 0.0 : XPRS_MINUSINFINITY; })
            .withUB([&](int s, int i, int j){ return k_i[i] + std::abs(d_s_i[s][i]); })
            .withName([](int s, int i, int j){ return xpress::format("o%s_%d_%d", j%2==0 ? "Pos" : "Neg", s, i); })
            .toArray();

        // end = std::chrono::high_resolution_clock::now();
        // saveTimeToInfoDf(infoDf, start, end, "Variables o (ms)");
        // start = std::chrono::high_resolution_clock::now();

        // Create unmet demand helper variables u
        std::vector<std::vector<std::vector<xpress::objects::Variable>>> u = prob
            .addVariables(NR_SCENARIOS, NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB([](int s, int i, int j) { return j%2 == 0 ? 0.0 : XPRS_MINUSINFINITY; })
            .withUB([&](int s, int i, int j){ return k_i[i] + std::abs(d_s_i[s][i]); })
            .withName([](int s, int i, int j){ return xpress::format("u%s_%d_%d", j%2==0 ? "Pos" : "Neg", s, i); })
            .toArray();

        // end = std::chrono::high_resolution_clock::now();
        // saveTimeToInfoDf(infoDf, start, end, "Variables u (ms)");
        // start = std::chrono::high_resolution_clock::now();

        // End of variable creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Variable Creation (ms)");

        /* CONSTRAINTS */
        std::cout << "CREATING CONSTRAINTS" << std::endl;

        // Count duration of constraint creation
        start = std::chrono::high_resolution_clock::now();

        // First Stage decision
        // prob.addConstraint(Utils::sum(x) <= NR_BIKES);
        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (x[i] <= k_i[i]).setName(xpress::format("Capacity%d", i));
        });


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

        // prob.addConstraints(NR_STATIONS, [&](int i) { return u[i][0] == 0.0; });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return o[i][0] == 0.0; });

        // prob.addConstraint(Utils::sum(during_day_net_customer_flows) == 0.0);
        // prob.addConstraint(Utils::sum(end_of_day_net_recourse_flows) == 0.0);

        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (end_of_day_net_recourse_flows[s][i] == during_day_net_customer_flows[s][i])
                    .setName(xpress::format("s%d_FlowCons_S%d", s, i));
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
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (u[s][i][0].maxOf(std::vector<Variable>{u[s][i][1]}, 0.0));
        });

        // o+[i] - o-[i] == -d_i_s[i] - (k[i] - x[i])
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (o[s][i][1] == -d_s_i[s][i] - (k_i[i] - x[i]));
        });
        prob.addConstraints(NR_SCENARIOS, NR_STATIONS, [&](int s, int i) {
            return (o[s][i][0].maxOf(std::vector<Variable>{o[s][i][1]}, 0.0));
        });

        // non-negativity constraints
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, 2, [&](int s, int i, int j) { return u[s][i][j] >= 0; });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, 2, [&](int s, int i, int j) { return o[s][i][j] >= 0; });
        // prob.addConstraints(NR_SCENARIOS, NR_STATIONS, NR_STATIONS, [&](int s, int i, int j) { return y[s][i][j] >= 0; });

        // End of constraint creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Constraint Creation (ms)");


        /* OBJECTIVE */
        std::cout << "CREATING OBJECTIVE" << std::endl;

        // Count duration of objective creation
        start = std::chrono::high_resolution_clock::now();

        LinExpression obj = LinExpression::create();
        for (int s=0; s<NR_SCENARIOS; s++) {
            for (int i=0; i<NR_STATIONS; i++) {
                for (int j=0; j<NR_STATIONS; j++) {
                    obj.addTerm(c_ij[i][j], y[s][i][j]);
                }
                obj.addTerm(q_i_1[i], u[s][i][0]);
                obj.addTerm(q_i_2[i], o[s][i][0]);
            }
        }

        // TODO: Use NR_SCENARIOS in P_s
        obj.addTerms(Utils::scalarProduct(x, c_i), NR_SCENARIOS);

        prob.setObjective(obj, xpress::ObjSense::Minimize);

        // End of objective creation
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Objective Creation (ms)");

        /* INSPECT, SOLVE & PRINT */

        int nrVariables = prob.getCols();

        int nrConstraints = prob.getRows();
        // std::vector<char> rowTypes = prob.getRowType(0, nrConstraints-1);
        // std::unordered_map<char, int> rowTypeCount;
        // for (const auto& elem : rowTypes) {  ++rowTypeCount[elem]; }
        // for (const auto& pair : rowTypeCount) { std::cout << "RowType: " << pair.first << ", Count: " << pair.second << std::endl; }
        
        int nrSets = prob.getSets();
        int nrVariablesInSets = prob.getSetMembers();
        
        infoDf.addColumn("NrVariables", std::vector<double>{double(nrVariables)});
        infoDf.addColumn("NrConstraints", std::vector<double>{double(nrConstraints)});
        infoDf.addColumn("NrOfSos1Constraints", std::vector<double>{double(nrSets)});
        infoDf.addColumn("NrVariablesInSos1Sets", std::vector<double>{double(nrVariablesInSets)});


        // Write the problem in LP format for manual inspection
        std::cout << "Writing the problem to 'SubProb.lp'" << std::endl;
        prob.writeProb("SubProb.lp", "l");

        // Solve the problem
        std::cout << "Solving the problem" << std::endl;
        // Count duration of Optimization
        start = std::chrono::high_resolution_clock::now();

        // Optimize
        if (SOLVE_LP_RELAXATION) prob.lpOptimize();
        else prob.optimize();

        // End of Optimization
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Optimization (ms)");


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

        for (int s=0; s<NR_SCENARIOS; s++) {
            // for (int i=0; i<NR_STATIONS; i++) {
            //     for (int j=0; j<NR_STATIONS; j++) {
            //         std::cout << y[s][i][j].getName() << " = " << y[s][i][j].getValue(sol) << "\t";
            //     }
            //     std::cout << std::endl;
            // }
            // std::cout << std::endl;

            for (int i=0; i<NR_STATIONS; i++) {
                int j = 0;
                for (int j=0; j<2; j++) {
                if (u[s][i][j].getValue(sol) > 0.1) {
                    std::cout << u[s][i][j].getName() << " = " << u[s][i][j].getValue(sol) << std::endl;
                }}
            }
            std::cout << std::endl;

            for (int i=0; i<NR_STATIONS; i++) {
                int j = 0;
                for (int j=0; j<2; j++) {
                if (o[s][i][j].getValue(sol) > 0.1) {
                    std::cout << o[s][i][j].getName() << " = " << o[s][i][j].getValue(sol) << std::endl;
                }}
            }
            std::cout << std::endl;
            std::cout << std::endl;
        }
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}


