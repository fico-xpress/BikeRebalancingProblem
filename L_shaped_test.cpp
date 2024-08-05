#include <xpress.hpp>
#include <stdexcept> // For throwing exceptions

using namespace xpress;
using namespace xpress::objects;


// TODO: Investigate effect of using x*y == 0.0 instead of max(0.0, x) operator
// TODO: Investigate effect of using partial integer variables instead of continuous or integer variables
// TODO: Reformulate using pos and neg slack constraints.
// TODO: Try making additional constraints for the LP-relaxation.

std::vector<double> myElementWiseAddition(std::vector<double> a, std::vector<double> b) {
    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}

double myScalarProduct(std::vector<double> a, std::vector<double> b) {
    double ans = 0.0;
    for (int i=0 ; i<a.size(); i++) {
        ans += a[i] * b[i];
    }
    return ans;
}

std::vector<std::vector<double>> multiplyMatrices(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Number of columns in A must be equal to the number of rows in B.");
    }

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


bool generateOptimalityCut(int t, std::vector<double>& masterSol_x_t, double theta, int NR_SCENARIOS,
                           std::vector<std::vector<double>>& d_s_i, std::vector<std::vector<double>>& q_s_i, std::vector<double>& p_s,
                           double& e_t, std::vector<double>& E_t) {

    // ################## Solving Sub Problems ######################

    int NR_FIRST_STAGE_VARIABES = masterSol_x_t.size();
    int NR_2ND_STAGE_CONSTRAINTS = 4;
    std::vector<std::vector<double>> h_k_constr(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));
    std::vector<std::vector<std::vector<double>>> T_k_x_constr(NR_SCENARIOS, std::vector<std::vector<double>>(NR_FIRST_STAGE_VARIABES, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS)));
    std::vector<std::vector<double>> pi_k_constr(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));


    for (int s=0; s<NR_SCENARIOS; s++) {
        XpressProblem subProb_s;
        /* VARIABLES */
        std::vector<Variable> y = subProb_s.addVariables(2).withLB(0).withName([s](int i){ return xpress::format("y_s%d_%d", s, i); }).toArray();
        /* CONSTRAINTS */
        T_k_x_constr[s] = {{-60, 0}, {0, -80}, {0, 0}, {0, 0}};
        h_k_constr[s] = {0, 0,  d_s_i[s][0],  d_s_i[s][1]};

        subProb_s.addConstraint(myScalarProduct(T_k_x_constr[s][0], masterSol_x_t) + 6*y[0] + 10*y[1] <= h_k_constr[s][0]);
        subProb_s.addConstraint(myScalarProduct(T_k_x_constr[s][1], masterSol_x_t) + 8*y[0] +  5*y[1] <= h_k_constr[s][1]);
        subProb_s.addConstraint(myScalarProduct(T_k_x_constr[s][2], masterSol_x_t) + 1*y[0] +  0*y[1] <= h_k_constr[s][2]);
        subProb_s.addConstraint(myScalarProduct(T_k_x_constr[s][3], masterSol_x_t) + 0*y[0] +  1*y[1] <= h_k_constr[s][3]);
        /* OBJECTIVE */
        subProb_s.setObjective(Utils::scalarProduct(y, q_s_i[s]), xpress::ObjSense::Minimize);

        /* INSPECT, SOLVE & PRINT */
        subProb_s.writeProb(xpress::format("SubProb_%d.lp", s), "l");
        subProb_s.optimize();

        // Check the solution status
        if (subProb_s.getSolStatus() != SolStatus::Optimal && subProb_s.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << subProb_s.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Retrieve the solution values
        std::cout << "Objective value = " << subProb_s.getObjVal() << std::endl;
        std::vector<double> subSol_y_s_t = subProb_s.getSolution(y);
        for (int i=0; i<2; i++) std::cout << y[i].getName() << " = " << subSol_y_s_t[i] << std::endl;

        pi_k_constr[s] = subProb_s.getDuals();
        std::vector<double> duals = subProb_s.getDuals();
        for (int i=0; i<duals.size(); i++) std::cout << duals[i] << std::endl;

        std::cout << std::endl;
    }

    // double e_t = 0.0;
    // std::vector<double> E_t(NR_FIRST_STAGE_VARIABES, 0.0);
    for (int s=0; s<NR_SCENARIOS; s++) {
        e_t += p_s[s] * myScalarProduct(pi_k_constr[s], h_k_constr[s]);
        std::vector<double> result = multiplyMatrices(std::vector<std::vector<double>>{pi_k_constr[s]}, T_k_x_constr[s])[0];
        for (int i=0 ; i<NR_FIRST_STAGE_VARIABES; i++) {
            E_t[i] += p_s[s] * result[i];
        }
    }
    std::cout << "e_" << t << " = " << e_t << std::endl;
    std::cout << "E_" << t << " = ";
    for (int i=0; i<E_t.size(); i++) std::cout << E_t[i] << "   ";
    std::cout << std::endl;

    double w_t = e_t - myScalarProduct(E_t, masterSol_x_t);
    std::cout << "w_" << t << " = " << w_t << std::endl;
    std::cout << "theta = " << theta << std::endl;
    std::cout << "w_" << t << " <= theta is " << (w_t <= theta) << std::endl;

    return w_t <= theta;
} 


void solveToOptimality(XpressProblem& masterProb, std::vector<Variable>& x, Variable& theta, double& e_t, std::vector<double>& E_t,
                       std::vector<std::vector<double>>& d_s_i, std::vector<std::vector<double>>& q_s_i, std::vector<double>& p_s) {
    int NR_FIRST_STAGE_VARIABES = x.size();
    int NR_SCENARIOS = p_s.size();

    int iter = 1;
    while (true) {
        iter++;
        // ################## Iteration t: Start of Solving Master Problem ######################

        masterProb.addConstraint(Utils::scalarProduct(x, E_t) + theta >= e_t);
        masterProb.optimize();

        // Check the solution status
        if (masterProb.getSolStatus() != SolStatus::Optimal && masterProb.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << masterProb.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Retrieve the solution values
        std::cout << "Master Objective = " << masterProb.getObjVal() << std::endl;
        std::vector<double> masterSol_x_t = masterProb.getSolution(x);
        double masterSol_theta_t = masterProb.getSolution(theta);

        for (int i=0; i<2; i++) std::cout << x[i].getName() << " = " << masterSol_x_t[i] << std::endl;
        std::cout << "theta = " << masterSol_theta_t << std::endl << std::endl;

        // ################## Iteration t: End of Solving Master Problem ######################
        // ################## Iteration t: Start of Solving Sub Problems ######################

        e_t = 0.0;
        std::fill(E_t.begin(), E_t.end(), 0.0);
        bool isOptimal = generateOptimalityCut(iter, masterSol_x_t, masterSol_theta_t, NR_SCENARIOS,
                                                d_s_i, q_s_i, p_s, e_t, E_t);
        
        if (isOptimal) {
            std::cout << "WOW optimality was found!" << std::endl;
            break;
        } else {
            std::cout << "\nAnother one" << std::endl;
        }
        // ################## Iteration t: End of Solving Sub Problems ######################
    }
}



int main() {
    try {

        // ################## Iteration 0: Start of Solving Master Problem ######################
        std::vector<double> c_i     = { 100, 150 };

        std::vector<double> p_s = { 0.4, 0.6};
        int NR_SCENARIOS = p_s.size();
        int NR_FIRST_STAGE_VARIABES = c_i.size();

        std::vector<std::vector<double>> d_s_i = {{ 500, 100 }, { 300, 300 }};
        std::vector<std::vector<double>> q_s_i = {{ -24, -28 }, { -28, -32 }};

        // Create a problem instance
        XpressProblem masterProb;
        // prob.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        /* VARIABLES */
        std::vector<Variable> x = masterProb.addVariables(2).withName([](int i){ return xpress::format("x_%d", i); }).toArray();
        /* CONSTRAINTS */
        masterProb.addConstraint(Utils::sum(x) <= 120);
        masterProb.addConstraint(x[0] >= 40);
        masterProb.addConstraint(x[1] >= 20);

        /* OBJECTIVE */
        masterProb.setObjective(Utils::scalarProduct(x, c_i), xpress::ObjSense::Minimize);

        /* INSPECT, SOLVE & PRINT */
        // masterProb.writeProb("SubProb.lp", "l");
        masterProb.optimize();

        // Check the solution status
        if (masterProb.getSolStatus() != SolStatus::Optimal && masterProb.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << masterProb.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Retrieve the solution values
        std::vector<double> masterSol_x_t = masterProb.getSolution(x);
        double masterSol_theta_t = XPRS_MINUSINFINITY;

        for (int i=0; i<2; i++) std::cout << x[i].getName() << " = " << masterSol_x_t[i] << std::endl;
        std::cout << "theta = " << masterSol_theta_t << std::endl << std::endl;
        std::cout << std::endl;

        // ################## Iteration 0: End of Solving Master Problem ######################
        // ################## Iteration 0: Start of Solving Sub Problems ######################

        double e_t = 0.0;
        std::vector<double> E_t(NR_FIRST_STAGE_VARIABES, 0.0);
        bool isOptimal = generateOptimalityCut(1, masterSol_x_t, masterSol_theta_t, NR_SCENARIOS,
                                               d_s_i, q_s_i, p_s, e_t, E_t);
        
        if (isOptimal) {
            std::cout << "WOW optimality was found!" << std::endl;
            return 0;
        } else {
            std::cout << "\nAnother one" << std::endl;
        }

        // ################## Iteration 0: End of Solving Sub Problems ######################
        // ################## Iteration 1: Start of Solving Master Problems ######################

        Variable theta = masterProb.addVariable(XPRS_MINUSINFINITY, XPRS_PLUSINFINITY, ColumnType::Continuous, "theta");
        // Add theta to the objective, with coefficient 1.0
        theta.chgObj(1.0);

        solveToOptimality(masterProb, x, theta, e_t, E_t, d_s_i, q_s_i, p_s);

        // ################## Iteration 1: End of Solving Master Problem ######################
    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}

