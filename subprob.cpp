#include <xpress.hpp>
#include <stdexcept> // For throwing exceptions

using namespace xpress;
using namespace xpress::objects;


// TODO: Investigate effect of using x*y == 0.0 instead of max(0.0, x) operator
// TODO: Investigate effect of using partial integer variables instead of continuous or integer variables
// TODO: Reformulate using pos and neg slack constraints.
// TODO: Try making additional constraints for the LP-relaxation.

int main() {
    try {
        int NR_STATIONS = 4;
        int NR_BIKES = -1;
        // std::vector<double> x_i   = { 2,  3,   16, 25 };
        std::vector<double> k_i   = { 10, 15,  20, 30 };
        std::vector<double> d_i_s = { 4,  -5,  8,  -8 };
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        std::vector<double> q_i_1(NR_STATIONS, 10);
        std::vector<double> q_i_2(NR_STATIONS, 10);

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
        std::vector<xpress::objects::Variable> u = prob.addVariables(NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withName([](int i){ return xpress::format("u_%d", i); })
            .toArray();

        // Create station overflow helper variables o
        std::vector<xpress::objects::Variable> o = prob.addVariables(NR_STATIONS)
            .withType(ColumnType::Continuous)
            .withLB(0)
            .withName([](int i){ return xpress::format("o_%d", i); })
            .toArray();

        /* CONSTRAINTS */

        // First Stage decision
        // prob.addConstraints(NR_STATIONS, [&](int i) {
        //     return (x[i] == x_i[i]).setName(xpress::format("FirstStage_%d", i));
        // });

        std::vector<LinExpression> end_of_day_net_recourse_flows(NR_STATIONS);
        std::vector<Expression> during_day_net_customer_flows(NR_STATIONS);

        for (int i=0; i<NR_STATIONS; i++) {
            LinExpression net_recourse_flow = LinExpression::create();
            for (int j=0; j<NR_STATIONS; j++) {
                net_recourse_flow.addTerm(y[i][j], 1).addTerm(y[j][i], -1);
            }
            end_of_day_net_recourse_flows[i] = net_recourse_flow;
            during_day_net_customer_flows[i] = -( d_i_s[i] -u[i] + o[i] );
        }

        prob.addConstraint(Utils::sum(during_day_net_customer_flows) == 0.0);
        // prob.addConstraint(Utils::sum(end_of_day_net_recourse_flows) == 0.0);

        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (end_of_day_net_recourse_flows[i] == during_day_net_customer_flows[i])
                    .setName(xpress::format("FlowCons_S%d", i));
        });

        // u[i] == max(0, d_i_s[i] - x[i])
        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (u[i] == Utils::max(d_i_s[i] - x[i], ConstantExpression(0.0)))
                    .setName(xpress::format("UnmetDem_S%d", i));
        });

        prob.addConstraints(NR_STATIONS, [&](int i) {
            return (o[i] == Utils::max(-d_i_s[i] - (k_i[i] - x[i]), ConstantExpression(0.0)))
                    .setName(xpress::format("Overflow_S%d", i));
        });

        // non-negativity constraints
        prob.addConstraints(NR_STATIONS, [&](int i) { return u[i] >= 0; });
        prob.addConstraints(NR_STATIONS, [&](int i) { return o[i] >= 0; });
        prob.addConstraints(NR_STATIONS, NR_STATIONS, [&](int i, int j) { return y[i][j] >= 0; });

        /* OBJECTIVE */

        LinExpression obj = LinExpression::create();
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                obj.addTerm(c_ij[i][j], y[i][j]);
            }
            obj.addTerm(q_i_1[i], u[i]);
            obj.addTerm(q_i_2[i], o[i]);
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

        for (Variable u_i : u) std::cout << u_i.getName() << " = " << u_i.getValue(sol) << std::endl;
        std::cout << std::endl;

        for (Variable o_i : o) std::cout << o_i.getName() << " = " << o_i.getValue(sol) << std::endl;
        std::cout << std::endl;
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}
