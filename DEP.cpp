#include <xpress.hpp>
#include <stdexcept>   // For throwing exceptions
#include <unordered_map>
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


int main() {

    try {
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

        std::vector<double> capacities_i = stationData.getColumn<double>("nbDocks");
        std::vector<std::vector<double>> demand_s_i = convertScenariosToMatrix(scenarios);

        std::cout << demand_s_i.size() << ", " << demand_s_i[0].size();

        return 0;



        int NR_STATIONS = 4;
        int NR_BIKES = 46;

        // std::vector<double> x_fixed = { 2,  3,   16, 25 };
        std::vector<double> k_i     = { 10, 15,  20, 30 };
        std::vector<double> d_i_s   = { 4,  -5,  9,  -8 };
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        std::vector<double> q_i_1(NR_STATIONS, 1);
        std::vector<double> q_i_2(NR_STATIONS, 1);

        // Create a problem instance
        XpressProblem prob;
        prob.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        /* VARIABLES */
        // Create first-stage variables x
        std::vector<xpress::objects::Variable> x = prob.addVariables(NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withName([](int i){ return xpress::format("x_%d", i); })
            .toArray();

        // Create recourse variables y
        std::vector<std::vector<xpress::objects::Variable>> y = prob.addVariables(NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withName([](int i, int j){ return xpress::format("y(%d,%d)", i, j); })
            .toArray();

        // Create unmet demand helper variables u
        std::vector<std::vector<xpress::objects::Variable>> u = prob.addVariables(NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int i, int j){ return k_i[i] + std::abs(d_i_s[i]); })
            .withName([](int i, int j){ return xpress::format("u%s_%d", j%2==0 ? "Pos" : "Neg", i); })
            .toArray();

        // Create station overflow helper variables o
        std::vector<std::vector<xpress::objects::Variable>> o = prob.addVariables(NR_STATIONS, 2)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withUB([&](int i, int j){ return k_i[i] + std::abs(d_i_s[i]); })
            .withName([](int i, int j){ return xpress::format("o%s_%d", j%2==0 ? "Pos" : "Neg", i); })
            .toArray();


        /* CONSTRAINTS */

        // First Stage decision
        prob.addConstraint(Utils::sum(x) <= NR_BIKES);
        // prob.addConstraints(NR_STATIONS, [&](int i) {
        //     return (x[i] == x_fixed[i]).setName(xpress::format("FirstStage_%d", i));
        // });

        std::vector<LinExpression> end_of_day_net_recourse_flows(NR_STATIONS);
        std::vector<Expression> during_day_net_customer_flows(NR_STATIONS);

        for (int i=0; i<NR_STATIONS; i++) {
            LinExpression net_recourse_flow = LinExpression::create();
            for (int j=0; j<NR_STATIONS; j++) {
                net_recourse_flow.addTerm(y[i][j], 1).addTerm(y[j][i], -1);
            }
            end_of_day_net_recourse_flows[i] = net_recourse_flow;
            during_day_net_customer_flows[i] = -( d_i_s[i] - u[i][0] + o[i][0] );
        }

        // prob.addConstraints(NR_STATIONS, [&](int i) { return u[i][0] == 0.0; });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return o[i][0] == 0.0; });

        // prob.addConstraint(Utils::sum(during_day_net_customer_flows) == 0.0);
        // prob.addConstraint(Utils::sum(end_of_day_net_recourse_flows) == 0.0);

        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (end_of_day_net_recourse_flows[i] == during_day_net_customer_flows[i])
                    .setName(xpress::format("FlowCons_S%d", i));
        });

        // Indicators with sos1
        prob.addConstraints(NR_STATIONS, [&](int i) { return SOS::sos1(u[i], std::vector<double>{0.0, 1.0}, xpress::format("sos1_u%d", i)); });
        prob.addConstraints(NR_STATIONS, [&](int i) { return SOS::sos1(o[i], std::vector<double>{0.0, 1.0}, xpress::format("sos1_o%d", i)); });
        // Indicators with bigM
        // std::vector<std::vector<xpress::objects::Variable>> indicators = prob.addVariables(2, NR_STATIONS).withType(ColumnType::Binary)
        //     .withName([](int i, int j){ return xpress::format("%sBool_%d", i%2==0 ? "u" : "o", j); }) .toArray();
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[0][i].ifThen(u[i][1] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[0][i].ifNotThen(u[i][0] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[1][i].ifThen(o[i][1] == 0.0); });
        // prob.addConstraints(NR_STATIONS, [&](int i) { return indicators[1][i].ifNotThen(o[i][0] == 0.0); });

        // u+[i] - u-[i] == d_i_s[i] - x[i]
        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (u[i][0] - u[i][1] == d_i_s[i] - x[i])
                    .setName(xpress::format("UnmetDem_S%d", i));
        });

        // o+[i] - o-[i] == -d_i_s[i] - (k[i] - x[i])
        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (o[i][0] - o[i][1] == -d_i_s[i] - (k_i[i] - x[i]))
                    .setName(xpress::format("Overflow_S%d", i));
        });

        // non-negativity constraints
        prob.addConstraints(NR_STATIONS, 2, [&](int i, int j) { return u[i][j] >= 0; });
        prob.addConstraints(NR_STATIONS, 2, [&](int i, int j) { return o[i][j] >= 0; });
        prob.addConstraints(NR_STATIONS, NR_STATIONS, [&](int i, int j) { return y[i][j] >= 0; });

        /* OBJECTIVE */

        LinExpression obj = LinExpression::create();
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                obj.addTerm(c_ij[i][j], y[i][j]);
            }
            obj.addTerm(q_i_1[i], u[i][0]);
            obj.addTerm(q_i_2[i], o[i][0]);
        }

        prob.setObjective(obj, xpress::ObjSense::Minimize);

        /* INSPECT, SOLVE & PRINT */

        // write the problem in LP format for manual inspection
        std::cout << "Writing the problem to 'SubProb.lp'" << std::endl;
        prob.writeProb("SubProb.lp", "l");

        // Solve the problem
        std::cout << "Solving the problem" << std::endl;
        prob.optimize();

        // Check the solution status
        if (prob.getSolStatus() != SolStatus::Optimal && prob.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << prob.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Print the solution to console (first set precision to e.g. 5)
        std::cout << std::endl << "*** Objective Value ***" << std::endl;
        std::cout << "Solution has objective value (profit) of " << prob.getObjVal() << std::endl;
        std::cout << std::endl << "*** Solution ***" << std::endl;

        // Retrieve the solution values in one go
        std::vector<double> sol = prob.getSolution();

        // Loop over the relevant variables and print their name and value
        for (Variable x_i : x) std::cout << x_i.getName() << " = " << x_i.getValue(sol) << std::endl;
        std::cout << std::endl;

        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                std::cout << y[i][j].getName() << " = " << y[i][j].getValue(sol) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<2; j++) {
                std::cout << u[i][j].getName() << " = " << u[i][j].getValue(sol) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;

        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<2; j++) {
                std::cout << o[i][j].getName() << " = " << o[i][j].getValue(sol) << "\t";
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}


