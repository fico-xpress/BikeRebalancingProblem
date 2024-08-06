#include <xpress.hpp>
#include <stdexcept> // For throwing exceptions
#include <omp.h>     // For parallelising for-loops in matrix multiplication


using namespace xpress;
using namespace xpress::objects;

std::vector<double> myElementWiseMultiplication(double a, std::vector<double>& b) {
    std::vector<double> ans(b.size());
    for (int i=0 ; i<b.size(); i++) {
        ans[i] = a * b[i];
    }
    return ans;
}

std::vector<double> myElementWiseMultiplication(std::vector<double>& a, std::vector<double>& b) {
    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] * b[i];
    }
    return ans;
}

std::vector<double> myElementWiseAddition(std::vector<double>& a, std::vector<double>& b) {
    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}

double myScalarProduct(std::vector<double>& a, std::vector<double>& b) {
    double ans = 0.0;
    for (int i=0 ; i<a.size(); i++) {
        ans += a[i] * b[i];
    }
    return ans;
}

std::vector<std::vector<double>> myMultiplyMatrices(std::vector<std::vector<double>>& A, std::vector<std::vector<double>>& B) {
    int rowsA = A.size();
    int colsA = A[0].size();
    int rowsB = B.size();
    int colsB = B[0].size();

    if (colsA != rowsB) {
        throw std::invalid_argument("Number of columns in A must be equal to the number of rows in B.");
    }

    std::vector<std::vector<double>> result(rowsA, std::vector<double>(colsB, 0));

    #pragma omp parallel for
    for (int i = 0; i < rowsA; ++i) {
        for (int j = 0; j < colsB; ++j) {
            for (int k = 0; k < colsA; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}


class TwoStage_LShapedMethod {
public:
    XpressProblem& masterProb;

    // Probability of each scenario s
    std::vector<double>& p_s;
    // Objective coefficients c for the first-stage decision variables x
    std::vector<double>& c_i;
    // Objective coefficients q for the second-stage decision variables y, for each scenario s
    std::vector<std::vector<double>>& q_s_i;
    // Right hand coefficients h_k (or h_s) for each 2nd-stage constraint and scenario
    std::vector<std::vector<double>>& d_s_i;

    // Constructor method: give all required coefficients / data
    TwoStage_LShapedMethod(XpressProblem& masterProb, std::vector<double>& p_s, std::vector<double>& c_i,
        std::vector<std::vector<double>>& q_s_i, std::vector<std::vector<double>>& d_s_i);

    void runLShapedMethod();
    std::vector<Variable>& getFirstStageDecisionVariables();


private:
    // First-stage decision variables
    std::vector<Variable> x;
    // Auxiliary variable in the master problem - necessary for the decomposition
    Variable theta;

    void makeInitialMasterProbFormulation();
    void solveMasterProb();
    void addOptimalityCutToMasterProb(std::vector<double>& E_t, double& e_t);
    bool generateOptimalityCut(std::vector<double>& E_t, double& e_t);

    int iter;

    int NR_SCENARIOS;
    int NR_1ST_STAGE_VARIABLES;
    int NR_2ND_STAGE_VARIABLES;
    int NR_2ND_STAGE_CONSTRAINTS;

    // To store the master problem's solution values of the variables x and theta
    std::vector<double> masterSol_x_t; //xSolValues;
    double masterSol_theta_t; //thetaSolValue;
};

// Constructor Method
TwoStage_LShapedMethod::TwoStage_LShapedMethod(XpressProblem& masterProb,
    std::vector<double>& p_s, std::vector<double>& c_i,
    std::vector<std::vector<double>>& q_s_i, std::vector<std::vector<double>>& d_s_i) 
     : masterProb(masterProb), p_s(p_s), c_i(c_i), q_s_i(q_s_i), d_s_i(d_s_i)
    {
        this->iter                      = 0;
        this->NR_SCENARIOS              = p_s.size();
        this->NR_1ST_STAGE_VARIABLES    = c_i.size();
        this->NR_2ND_STAGE_VARIABLES    = q_s_i.size();
        this->NR_2ND_STAGE_CONSTRAINTS  = d_s_i.size();
}

void TwoStage_LShapedMethod::runLShapedMethod() {

    makeInitialMasterProbFormulation();

    // ################## Iteration 0: Start of Solving Master Problem ######################
    std::cout << std::endl << "STARTING ITERATION " << iter << std::endl;
    solveMasterProb();
    // ################## Iteration 0: End of Solving Master Problem ######################
    // ################## Iteration 0: Start of Solving Sub Problems ######################

    double e_t = 0.0;
    std::vector<double> E_t(NR_1ST_STAGE_VARIABLES, 0.0);
    bool masterIsOptimal = this->generateOptimalityCut(E_t, e_t);

    if (masterIsOptimal) {
        std::cout << std::endl << "Optimality was found!" << std::endl;
        return;
    }

    this->theta = masterProb.addVariable(XPRS_MINUSINFINITY, XPRS_PLUSINFINITY, ColumnType::Continuous, "theta");
    // Add theta to the objective, with coefficient 1.0
    theta.chgObj(1.0);

    // ################## Iteration 0: End of Solving Sub Problems ######################
    // ################## Perform the Rest of the Iterations ######################

    while (true) {
        iter++;
        std::cout << "STARTING ITERATION " << iter << std::endl;

        addOptimalityCutToMasterProb(E_t, e_t);
        solveMasterProb();

        e_t = 0.0;
        std::fill(E_t.begin(), E_t.end(), 0.0);
        bool masterIsOptimal = this->generateOptimalityCut(E_t, e_t);

        if (masterIsOptimal) {
            std::cout << std::endl << "Optimality was found!" << std::endl;
            return;
        }
    }
}

std::vector<Variable>& TwoStage_LShapedMethod::getFirstStageDecisionVariables() {
    return x;
}

void TwoStage_LShapedMethod::makeInitialMasterProbFormulation() {
    /* VARIABLES */
    this->x = masterProb.addVariables(NR_1ST_STAGE_VARIABLES).withName([](int i){ return xpress::format("x_%d", i); }).toArray();

    /* CONSTRAINTS */
    masterProb.addConstraint(Utils::sum(x) <= 120);
    masterProb.addConstraint(x[0] >= 40);
    masterProb.addConstraint(x[1] >= 20);

    /* OBJECTIVE */
    masterProb.setObjective(Utils::scalarProduct(x, c_i), xpress::ObjSense::Minimize);
}

void TwoStage_LShapedMethod::solveMasterProb() {
    /* INSPECT */
    // masterProb.writeProb("MasterProb.lp", "l");

    /* SOLVE */
    masterProb.optimize();

    // Check the solution status
    if (masterProb.getSolStatus() != SolStatus::Optimal && masterProb.getSolStatus() != SolStatus::Feasible) {
        std::ostringstream oss; oss << masterProb.getSolStatus(); // Convert xpress::SolStatus to String
        throw std::runtime_error("Optimization failed with status " + oss.str());
    }

    // Retrieve the solution values
    this->masterSol_x_t = masterProb.getSolution(this->x);

    // If the theta-variable has not yet been added to the masterProb, its value is Minus Infinity
    if (masterProb.getOriginalCols() == x.size()) {
        this->masterSol_theta_t = XPRS_MINUSINFINITY;
    } else {
        this->masterSol_theta_t = masterProb.getSolution(this->theta);
    }

    /* PRINT */
    std::cout << "\tMaster Problem Solution" << std::endl;
    std::cout << "\t\tMaster Objective = " << masterProb.getObjVal() << std::endl;
    for (int i=0; i<2; i++) std::cout << "\t\t" << x[i].getName() << " = " << masterSol_x_t[i] << std::endl;
    std::cout << "\t\ttheta = " << (masterSol_theta_t == XPRS_MINUSINFINITY ? "MINUS INFINITY" : std::to_string(masterSol_theta_t)) << std::endl;
    std::cout << std::endl;
}

void TwoStage_LShapedMethod::addOptimalityCutToMasterProb(std::vector<double>& E_t, double& e_t) {
    masterProb.addConstraint(Utils::scalarProduct(x, E_t) + theta >= e_t);
    std::cout << "\tAdding constraint: " << (Utils::scalarProduct(x, E_t) + theta).toString() << " >= " << e_t << std::endl << std::endl;
}

bool TwoStage_LShapedMethod::generateOptimalityCut(std::vector<double>& E_t, double& e_t) {

    // ################## Solving Sub Problems ######################

    std::vector<std::vector<double>> h_s_constr(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));
    std::vector<std::vector<std::vector<double>>> T_s_x_constr(NR_SCENARIOS, std::vector<std::vector<double>>(NR_1ST_STAGE_VARIABLES, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS)));
    std::vector<std::vector<double>> pi_s_constr(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));


    for (int s=0; s<NR_SCENARIOS; s++) {
        XpressProblem subProb_s;
        /* VARIABLES */
        std::vector<Variable> y = subProb_s.addVariables(2).withLB(0).withName([s](int i){ return xpress::format("y_s%d_%d", s, i); }).toArray();
        /* CONSTRAINTS */
        T_s_x_constr[s] = {{-60, 0}, {0, -80}, {0, 0}, {0, 0}};
        h_s_constr[s] = {0, 0,  d_s_i[s][0],  d_s_i[s][1]};
        NR_2ND_STAGE_CONSTRAINTS = h_s_constr[s].size();

        subProb_s.addConstraint(myScalarProduct(T_s_x_constr[s][0], masterSol_x_t) + 6*y[0] + 10*y[1] <= h_s_constr[s][0]);
        subProb_s.addConstraint(myScalarProduct(T_s_x_constr[s][1], masterSol_x_t) + 8*y[0] +  5*y[1] <= h_s_constr[s][1]);
        subProb_s.addConstraint(myScalarProduct(T_s_x_constr[s][2], masterSol_x_t) + 1*y[0] +  0*y[1] <= h_s_constr[s][2]);
        subProb_s.addConstraint(myScalarProduct(T_s_x_constr[s][3], masterSol_x_t) + 0*y[0] +  1*y[1] <= h_s_constr[s][3]);
        /* OBJECTIVE */
        subProb_s.setObjective(Utils::scalarProduct(y, q_s_i[s]), xpress::ObjSense::Minimize);

        /* INSPECT, SOLVE & PRINT */
        // subProb_s.writeProb(xpress::format("SubProb_%d.lp", s), "l");
        subProb_s.optimize();

        // Check the solution status
        if (subProb_s.getSolStatus() != SolStatus::Optimal && subProb_s.getSolStatus() != SolStatus::Feasible) {
            std::ostringstream oss; oss << subProb_s.getSolStatus(); // Convert xpress::SolStatus to String
            throw std::runtime_error("Optimization failed with status " + oss.str());
        }

        // Retrieve the solution values
        std::cout << "\tScenario " << s << ": Sub Problem Solution" << std::endl;
        std::cout << "\t\tObjective value = " << subProb_s.getObjVal() << std::endl;
        std::vector<double> subSol_y_s_t = subProb_s.getSolution(y);
        for (int i=0; i<2; i++) std::cout << "\t\t" << y[i].getName() << " = " << subSol_y_s_t[i] << std::endl;

        pi_s_constr[s] = subProb_s.getDuals();
        std::cout << "\t\tpi_s" << s << " = ";
        std::vector<double> duals = subProb_s.getDuals();
        for (int i=0; i<duals.size(); i++) std::cout << duals[i] << ",  ";
        std::cout << std::endl << std::endl;
    }

    for (int s=0; s<NR_SCENARIOS; s++) {
        e_t += p_s[s] * myScalarProduct(pi_s_constr[s], h_s_constr[s]);
        std::vector<double> result = myMultiplyMatrices(std::vector<std::vector<double>>{pi_s_constr[s]}, T_s_x_constr[s])[0];
        // E_t = myElementWiseAddition(E_t, myElementWiseMultiplication(p_s[s], result));
        for (int i=0 ; i<NR_1ST_STAGE_VARIABLES; i++) {
            E_t[i] += p_s[s] * result[i];
        }
    }
    std::cout << "\tGenerated Cut:" << std::endl;
    std::cout << "\t\te_" << iter << " = " << e_t << std::endl;
    std::cout << "\t\tE_" << iter << " = ";
    for (int i=0; i<E_t.size(); i++) std::cout << E_t[i] << ",  ";
    std::cout << std::endl << std::endl;

    double w_t = e_t - myScalarProduct(E_t, masterSol_x_t);
    std::cout << "\tw_" << iter << " = " << w_t << std::endl;
    std::cout << "\ttheta = " << masterSol_theta_t << std::endl;
    std::cout << "\tw_" << iter << " <= theta is " << (w_t <= masterSol_theta_t ? "True" : "False") << std::endl;
    std::cout << std::endl;
    return w_t <= masterSol_theta_t;
}



int main() {
    try {
        // Initialize coefficients
        std::vector<double> c_i = { 100, 150 };
        std::vector<double> p_s = { 0.4, 0.6 };
        std::vector<std::vector<double>> d_s_i = {{ 500, 100 }, { 300, 300 }};
        std::vector<std::vector<double>> q_s_i = {{ -24, -28 }, { -28, -32 }};

        // Create a problem instance
        XpressProblem masterProb;
        // masterProb.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        // Initialize Two-Stage Stochastic Problem solver
        TwoStage_LShapedMethod tssp_solver = TwoStage_LShapedMethod(masterProb, p_s, c_i, q_s_i, d_s_i);

        // Solve the TSSP
        tssp_solver.runLShapedMethod();
        std::vector<Variable>& x = tssp_solver.getFirstStageDecisionVariables();

        // Print optimal first-stage decisions and objective
        std::cout << std::endl << "*** OPTIMAL SOLUTION FOUND ***" << std::endl;
        std::cout << "Master Objective = " << masterProb.getObjVal() << std::endl;
        std::cout << "First Stage Decision Variables:" << std::endl;
        std::vector<double> solution = masterProb.getSolution();
        for (int i=0; i<2; i++) {
            std::cout << "\t" << x[i].getName() << " = " << x[i].getValue(solution) << std::endl;
        }

    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}

