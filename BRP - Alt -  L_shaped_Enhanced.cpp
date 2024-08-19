#include <xpress.hpp>
#include <stdexcept> // For throwing exceptions
#include <unordered_map>
#include <chrono>   // For timekeeping
#include <numeric>  // For std::iota() function
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

using TimeDataType = std::chrono::time_point<std::chrono::high_resolution_clock>;
void saveTimeToInfoDf(DataFrame& infoDf, TimeDataType start, TimeDataType end, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    long long duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();
    std::cout << "'" << columnName << "' took " << duration << "ms (" << duration/1000.0 << "s)" << std::endl;

    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<long long>{duration});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_L_Enhanced_SingleU.csv";
        infoDf.toCsv(fileName.str());
    }
}

void saveDoubleToInfoDf(DataFrame& infoDf, double value, std::string columnName, int NR_STATIONS, int NR_SCENARIOS) {
    if (!infoDf.hasColumnName(columnName)) {
        infoDf.addColumn(columnName, std::vector<double>{value});

        std::ostringstream fileName;
        fileName << "./time_data/B=" << NR_STATIONS << "_S=" << NR_SCENARIOS << "_BRP_L_Enhanced_SingleU.csv";
        infoDf.toCsv(fileName.str());
    }
}


double mySum(std::vector<double> a) {
    double ans = 0.0;
    for (double val : a) ans += val;
    return ans;
}

std::vector<double> myElementWiseMultiplication(double a, std::vector<double>& b) {
    std::vector<double> ans(b.size());
    for (int i=0 ; i<b.size(); i++) {
        ans[i] = a * b[i];
    }
    return ans;
}

std::vector<double> myElementWiseMultiplication(std::vector<double>& a, std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] * b[i];
    }
    return ans;
}

std::vector<double> myElementWiseAddition(std::vector<double>& a, std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

    std::vector<double> ans(a.size());
    for (int i=0 ; i<a.size(); i++) {
        ans[i] = a[i] + b[i];
    }
    return ans;
}

double myScalarProduct(std::vector<double>& a, std::vector<double>& b) {
    if (a.size() != b.size()) throw std::invalid_argument("Vectors a and b have different lengths");

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


// Forward declaration of TwoStage_LShapedMethod class such that the BRP_SubProblem knows it exists
class TwoStage_LShapedMethod;


class BRP_SubProblem {
public:
    std::unique_ptr<XpressProblem> subProbPtr;
    TwoStage_LShapedMethod* masterProbSolver;
    int s;
    int& NR_STATIONS;

    // 2nd-stage decision variables
    std::vector<std::vector<Variable>> y;
    std::vector<Variable> u;
    std::vector<Variable> o;

    BRP_SubProblem(TwoStage_LShapedMethod* masterProbSolver, std::unique_ptr<XpressProblem> subProbPtr, int subProbIndex);
    void makeInitialSubProbFormulation();
    void updateFirstStageVariableValues();
    std::vector<double> computeNewRightHandSides();
    void BRP_SubProblem::solveSubProblem();
};



class TwoStage_LShapedMethod {
     // Such that the BRP_SubProblem class can access TwoStage_LShapedMethod's private members
    friend class BRP_SubProblem;

public:
    XpressProblem& masterProb;

    // Probability of each scenario s
    std::vector<double>& p_s;

    // Objective coefficients c for each first-stage decision variables x_i
    std::vector<double>& c_i;
    // Right-hand coefficients b for each first-stage constraint j
    std::vector<double>& b_i;

    // Objective coefficients for each second-stage decision variable y_ij
    std::vector<std::vector<double>> c_ij;
    // Objective coefficients for each second-stage decision variable o_i, u_i
    std::vector<double> q_i_1, q_i_2;
    // Right hand coefficients h for some 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<double>> d_s_i;

    // Constructor method: give all required coefficients / data
    TwoStage_LShapedMethod(XpressProblem& masterProb, std::vector<double>& c_i, std::vector<double>& b_i, 
        std::vector<double>& p_s, std::vector<std::vector<double>>& c_ij, std::vector<double>& q_i_1,
        std::vector<double>& q_i_2, std::vector<std::vector<double>>& d_s_i);

    void runLShapedMethod(int NR_BIKES, bool verbose);
    std::vector<Variable>& getFirstStageDecisionVariables();
    double getExpectedSecondStageCost();
    double getNumberOfIterations() {return iter;};


private:
    // First-stage decision variables
    std::vector<Variable> x;
    // Auxiliary variables in the master problem - necessary for the decomposition
    std::vector<Variable> theta;

    void makeInitialSubProbFormulation(int s);
    void makeInitialMasterProbFormulation(int NR_BIKES);
    void solveMasterProb(bool solveRelaxation);
    void addOptimalityCutToMasterProb(int s, std::vector<double>& E_t, double& e_t);
    bool generateOptimalityCut(std::vector<double>& E_t, double& e_t);

    int iter;
    bool verbose;

    int NR_SCENARIOS;
    int NR_1ST_STAGE_VARIABLES;
    int NR_1ST_STAGE_CONSTRAINTS;
    int NR_2ND_STAGE_VARIABLES;
    int NR_2ND_STAGE_CONSTRAINTS;

    int NR_STATIONS;

    // To store the master problem's solution values of the variables x and theta
    std::vector<double> masterSol_x_t; //xSolValues;
    double masterSol_theta_t; //thetaSolValue;


    // To store a subproblem for each scenario
    std::vector<BRP_SubProblem> savedSubproblems;

    // To store the right hand coefficients h for each 2nd-stage constraint j, for each scenario s
    std::vector<std::vector<double>> h_s_j;
    // To store the constraint coefficients T for each 1st-stage variable x_i, for each 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<std::vector<double>>> T_s_j_i;
};

// Constructor Method
TwoStage_LShapedMethod::TwoStage_LShapedMethod(XpressProblem& masterProb,
        std::vector<double>& c_i, std::vector<double>& b_i, std::vector<double>& p_s,
        std::vector<std::vector<double>>& c_ij, std::vector<double>& q_i_1,
        std::vector<double>& q_i_2, std::vector<std::vector<double>>& d_s_i) 
     : masterProb(masterProb), c_i(c_i), b_i(b_i), p_s(p_s), c_ij(c_ij), 
                               q_i_1(q_i_1), q_i_2(q_i_2), d_s_i(d_s_i)
    {
        this->iter                      = 0;
        this->NR_SCENARIOS              = p_s.size();
        this->NR_1ST_STAGE_VARIABLES    = c_i.size();
        this->NR_1ST_STAGE_CONSTRAINTS  = b_i.size();
        NR_STATIONS = NR_1ST_STAGE_VARIABLES;

        this->NR_2ND_STAGE_VARIABLES    = NR_STATIONS * NR_STATIONS + 2 * NR_STATIONS;
        this->NR_2ND_STAGE_CONSTRAINTS  = 3 * NR_STATIONS;

        // Initialize vector of subproblems (one for each scenario)
        for (int s=0; s<NR_SCENARIOS; s++) {
            // To make sure the created XpressProblem-object lives for longer than just this for-loop, we need to turn
            // the XpressProblem into a unique_ptr such that we are able to transfer ownership of the object to the BRP_SubProblem subclass
            std::unique_ptr<XpressProblem> subProbPtr = std::make_unique<XpressProblem>();
            // Initialize BRP_SubProblem with transferred ownership of the subProbPtr
            savedSubproblems.push_back(BRP_SubProblem(this, std::move(subProbPtr), s));
        }

        // To store the right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        h_s_j = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));
        // To store the constraint coefficients T for each 1st-stage variable x_i, for each 2nd-stage constraints j, for each scenario s
        T_s_j_i = std::vector<std::vector<std::vector<double>>>(NR_SCENARIOS, std::vector<std::vector<double>>(NR_2ND_STAGE_CONSTRAINTS, 
                                                                                          std::vector<double>(NR_1ST_STAGE_VARIABLES, 0.0)));
}

void TwoStage_LShapedMethod::runLShapedMethod(int NR_BIKES, bool verbose) {
    this->verbose = verbose;
    std::cout << std::endl << "STARTING ITERATION " << iter << std::endl;

    /* ################## Iteration 0: Model the Master Problem ###################### */
    makeInitialMasterProbFormulation(NR_BIKES);

    /* ################## Iteration 0: Start of Solving Master Problem ###################### */
    solveMasterProb(true);

    /* ################## Iteration 0: Model all the Sub Problems ###################### */
    for (int s=0; s<NR_SCENARIOS; s++) {
        savedSubproblems[s].makeInitialSubProbFormulation();
    }

    /* ################## Iteration 0: Add new Variable to Master Problem ###################### */
    this->theta = masterProb.addVariables(NR_SCENARIOS)
        // .withLB(XPRS_MINUSINFINITY).withUB(XPRS_PLUSINFINITY)
        .withType(ColumnType::Continuous).toArray();
    // Add theta to the objective, with coefficient 1.0
    for (int s=0; s<NR_SCENARIOS; s++) {
        theta[s].chgObj(1.0);
    }

    /* ################## Iteration 0: Start of Solving Sub Problems ###################### */
    double e_t = 0.0;
    std::vector<double> E_t(NR_1ST_STAGE_VARIABLES, 0.0);
    bool masterIsOptimal = this->generateOptimalityCut(E_t, e_t);

    if (masterIsOptimal) {
        std::cout << std::endl << "Optimality was found!" << std::endl;
        return;
    }

    /* ########################### End of Iteration 0 ############################# */
    /* ################## Perform the Rest of the Iterations ###################### */

    while (true) {
        iter++;
        std::cout << "STARTING ITERATION " << iter << std::endl;

        // addOptimalityCutToMasterProb(E_t, e_t);
        solveMasterProb(true);

        e_t = 0.0;
        std::fill(E_t.begin(), E_t.end(), 0.0);
        bool masterIsOptimal = this->generateOptimalityCut(E_t, e_t);

        if (masterIsOptimal) {
            break;
        }
    }
    std::cout << std::endl << "Optimality was found!" << std::endl;
    solveMasterProb(false);
}

std::vector<Variable>& TwoStage_LShapedMethod::getFirstStageDecisionVariables() {
    return x;
}

double TwoStage_LShapedMethod::getExpectedSecondStageCost() {
    return mySum(masterProb.getSolution(theta));
}

void TwoStage_LShapedMethod::makeInitialMasterProbFormulation(int NR_BIKES) {
    /* VARIABLES */
    this->x = masterProb.addVariables(NR_1ST_STAGE_VARIABLES)
        .withType(ColumnType::Integer)
        .withName([](int i){ return xpress::format("x_%d", i); })
        .toArray();

    /* CONSTRAINTS */
    masterProb.addConstraint(Utils::sum(x) == NR_BIKES).setName("Nr bikes constraint");

    masterProb.addConstraints(NR_1ST_STAGE_VARIABLES, [&](int i) {
        return (x[i] <= b_i[i]).setName(xpress::format("Capacity%d", i));
    });

    /* OBJECTIVE */
    masterProb.setObjective(Utils::scalarProduct(x, c_i), xpress::ObjSense::Minimize);
}

void TwoStage_LShapedMethod::solveMasterProb(bool solveRelaxation) {
    /* INSPECT */
    // masterProb.writeProb(xpress::format("MasterProb_%d.lp", iter), "l");

    /* SOLVE */
    if (solveRelaxation) {
        masterProb.lpOptimize(); // NOTE: solve LP-relaxation
    } else {
        masterProb.setMipRelStop(0.01);
        masterProb.optimize();
    }

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
        this->masterSol_theta_t = mySum(masterProb.getSolution(this->theta));
    }

    /* PRINT */
    std::cout << "\tMaster Problem Solved" << std::endl;
    std::cout << "\t\tMaster Objective = " << masterProb.getObjVal() << std::endl;
    if (verbose) {
        for (int i=0; i<x.size(); i++) std::cout << "\t\t" << x[i].getName() << " = " << masterSol_x_t[i] << std::endl;
        std::cout << "\t\ttheta = " << (masterSol_theta_t == XPRS_MINUSINFINITY ? "MINUS INFINITY" : std::to_string(masterSol_theta_t)) << std::endl;
        std::cout << std::endl;
    }
}

void TwoStage_LShapedMethod::addOptimalityCutToMasterProb(int s, std::vector<double>& E_t, double& e_t) {
    if (E_t.size() != x.size()) throw std::invalid_argument("Vectors E_t and x have different lengths");

    masterProb.addConstraint(Utils::scalarProduct(x, E_t) + theta[s] >= e_t);

    // std::cout << "\tAdding constraint" << std::endl;
    if (verbose) {
        std::cout << "\t\t" << (Utils::scalarProduct(x, E_t) + theta[s]).toString() << " >= " << e_t << std::endl << std::endl;
    }
}

bool TwoStage_LShapedMethod::generateOptimalityCut(std::vector<double>& E_t, double& e_t) {

    // ################## Solving Sub Problems ######################

    // To store the dual values pi for each 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<double>> pi_s_j(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));

    for (int s=0; s<NR_SCENARIOS; s++) {
        BRP_SubProblem& subProbSolver = savedSubproblems[s];
        // Remake subProb from scratch:
        // std::unique_ptr<XpressProblem> subProbPtr = std::make_unique<XpressProblem>();
        // BRP_SubProblem& subProb_solver = BRP_SubProblem(this, std::move(subProbPtr), s);
        // subProb_solver.makeInitialSubProbFormulation();

        subProbSolver.updateFirstStageVariableValues();
        subProbSolver.solveSubProblem();

        // Get dual values
        pi_s_j[s] = subProbSolver.subProbPtr->getDuals();
        if (pi_s_j[s].size() != NR_2ND_STAGE_CONSTRAINTS) throw std::invalid_argument("Disable presolve please, thanks");

        if (verbose) {
            std::cout << "\t\tpi_s" << s << " = ";
            for (int i=0; i<pi_s_j[s].size(); i++) std::cout << pi_s_j[s][i] << ",  ";
            std::cout << std::endl << std::endl;
        }
    }
    std::cout << "\tSolved All Sub Problems" << std::endl;

    for (int s=0; s<NR_SCENARIOS; s++) {
        double e_t_s = p_s[s] * myScalarProduct(pi_s_j[s], h_s_j[s]);

        std::vector<double> result = myMultiplyMatrices(std::vector<std::vector<double>>{pi_s_j[s]}, T_s_j_i[s])[0];
        std::vector<double> E_t_s = myElementWiseMultiplication(p_s[s], result);
        // E_t = myElementWiseAddition(E_t, myElementWiseMultiplication(p_s[s], result));
        // for (int i=0 ; i<NR_1ST_STAGE_VARIABLES; i++) {
        //     E_t[i] += p_s[s] * result[i];
        // }
        addOptimalityCutToMasterProb(s, E_t_s, e_t_s);

        E_t = myElementWiseAddition(E_t, E_t_s);
        e_t += e_t_s;//p_s[s] * myScalarProduct(pi_s_j[s], h_s_j[s]);
    }
    // masterProb.addConstraint(Utils::scalarProduct(x, E_t) + Utils::sum(theta) >= e_t);

    double w_t = e_t - myScalarProduct(E_t, masterSol_x_t);
    double epsilon = 0.01;
    double gap = (w_t - masterSol_theta_t)/std::abs(masterSol_theta_t);

    // Print some information
    std::cout << "\tGenerated Cut" << std::endl;
    if (verbose) {
        std::cout << "\t\te_" << iter << " = " << e_t << std::endl;
        std::cout << "\t\tE_" << iter << " = ";
        for (int i=0; i<NR_1ST_STAGE_VARIABLES; i++) std::cout << E_t[i] << ",  ";
        std::cout << std::endl;
    }
    std::cout  << std::endl;

    // Print some more information
    std::cout << "\tw_" << iter << " = " << w_t << std::endl;
    std::cout << "\ttheta = " << masterSol_theta_t << std::endl;
    std::cout << "\tw_" << iter << " <= theta is " << (w_t <= masterSol_theta_t ? "True" : "False") << std::endl;
    std::cout << "\tgap_" << iter << " = " << gap << " <= eps is " << (gap <= epsilon ? "True" : "False") << std::endl;
    std::cout << std::endl;

    return gap <= epsilon;
}





// DONE: Solve subproblems in an array for next iteration and modify in the next iteration
//          - YES (one problem per scenario, same |S| problems for each iteration)

// TODO: Callbacks to generate optimality cuts for each integer feasible solution:
//          - dont use `addConstraint()` in the callback  (this resets solver state)
//          - instead, use `addCut()` in the callback (because why not, decrease search space whenever you can), 
//            and also add to constraint pool for `addConstraints()` later
// TODO: Warm start for subproblems
//          - saving the basis of a node deep into the branch-and-bound tree, does not help very much
//              --> because it was found after adding a lot of constraints
//              --> so you modified the problem, which is now lost. 
//          - save basis of iteration i's MIP's LP-relaxation --> used as basis for iteration i+1's LP-relaxation
//          - loadmipsol () used in heuristics for the MIP
// TODO: Parrallellize subproblem solving
//          - first thing to do lol
// TODO: Python?
// TODO: multi-cut?



// Constructor Method of BRP_SubProblem class, taking unique ownership of the subProbPtr
BRP_SubProblem::BRP_SubProblem(TwoStage_LShapedMethod* masterProbSolver, std::unique_ptr<XpressProblem> subProbPtr, int subProbIndex) 
     : masterProbSolver(masterProbSolver), subProbPtr(std::move(subProbPtr)), NR_STATIONS(masterProbSolver->NR_STATIONS)
    {
        this->s = subProbIndex;
}

void BRP_SubProblem::makeInitialSubProbFormulation() {
    // subProbPtr->callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

    /* VARIABLES */
    this->y = subProbPtr->addVariables(NR_STATIONS, NR_STATIONS)
        .withType(ColumnType::Continuous)
        .withName([&](int i, int j){ return xpress::format("s%d_y(%d,%d)", s, i, j); })
        .toArray();

    this->u = subProbPtr->addVariables(NR_STATIONS)
        .withType(ColumnType::Continuous)
        .withName([&](int i){ return xpress::format("s%d_u(%d)", s, i); })
        .toArray();

    this->o = subProbPtr->addVariables(NR_STATIONS)
        .withType(ColumnType::Continuous)
        .withName([&](int i){ return xpress::format("s%d_o(%d)", s, i); })
        .toArray();
    
    /* CONSTRAINTS */
    for (int i=0; i<NR_STATIONS; i++) {
        // u_i[i] >= std::max(0.0, d_s_i[s][i] - masterSol_x_t[i]);
        // o_i[i] >= std::max(0.0, -d_s_i[s][i] - (b_i[i] - masterSol_x_t[i]));
        masterProbSolver->h_s_j[s][0*NR_STATIONS+i]      = - masterProbSolver->d_s_i[s][i];
        masterProbSolver->h_s_j[s][1*NR_STATIONS+i]      =   masterProbSolver->d_s_i[s][i];
        masterProbSolver->h_s_j[s][2*NR_STATIONS+i]      = - masterProbSolver->d_s_i[s][i] - masterProbSolver->b_i[i];
        masterProbSolver->T_s_j_i[s][0*NR_STATIONS+i][i] = 0.0;
        masterProbSolver->T_s_j_i[s][1*NR_STATIONS+i][i] = 1.0;
        masterProbSolver->T_s_j_i[s][2*NR_STATIONS+i][i] = -1.0;
    }
    
    std::vector<LinExpression> end_of_day_net_recourse_flows(NR_STATIONS);

    for (int i=0; i<NR_STATIONS; i++) {
        LinExpression net_recourse_flow = LinExpression::create();
        for (int j=0; j<NR_STATIONS; j++) {
            net_recourse_flow.addTerm(y[i][j], 1).addTerm(y[j][i], -1);
        }
        end_of_day_net_recourse_flows[i] = net_recourse_flow;
        // during_day_net_customer_flows[i] = -( d_s_i[s][i] - u[i] + o[i] );
    }

    subProbPtr->addConstraints(NR_STATIONS, [&](int j) {
        int offset = 0*NR_STATIONS;
        return (end_of_day_net_recourse_flows[j] == u[j] - o[j] + masterProbSolver->h_s_j[s][offset+j]
                 - myScalarProduct(masterProbSolver->T_s_j_i[s][offset+j], masterProbSolver->masterSol_x_t));
                // .setName(xpress::format("FlowCons_S%d", j));
    });
    // u_i[i] >= std::max(0.0, d_s_i[s][i] - masterSol_x_t[i]);
    subProbPtr->addConstraints(NR_STATIONS, [&](int j) {
        int offset = 1*NR_STATIONS;
        return (u[j] >= masterProbSolver->h_s_j[s][offset+j]
                 - myScalarProduct(masterProbSolver->T_s_j_i[s][offset+j], masterProbSolver->masterSol_x_t));
                // .setName(xpress::format("underflow_%d", j));
    });
    // o_i[i] >= std::max(0.0, -d_s_i[s][i] - (b_i[i] - masterSol_x_t[i]));
    subProbPtr->addConstraints(NR_STATIONS, [&](int j) {
        int offset = 2*NR_STATIONS;
        return (o[j] >= masterProbSolver->h_s_j[s][offset+j]
                 - myScalarProduct(masterProbSolver->T_s_j_i[s][offset+j], masterProbSolver->masterSol_x_t));
                // .setName(xpress::format("overflow_%d", j));
    });

    // std::cout << "\tBuilt sub problem constraints" << std::endl;

    /* OBJECTIVE */
    LinExpression objective = LinExpression::create();
    for (int i=0; i<NR_STATIONS; i++) {
        for (int j=0; j<NR_STATIONS; j++) {
            objective.addTerm(masterProbSolver->c_ij[i][j], y[i][j]);
        }
        objective.addTerm(masterProbSolver->q_i_1[i], u[i]);
        objective.addTerm(masterProbSolver->q_i_2[i], o[i]);
    }
    subProbPtr->setObjective(objective, xpress::ObjSense::Minimize);
}

std::vector<double> BRP_SubProblem::computeNewRightHandSides() {
    std::vector<double> rhsCoeffs(masterProbSolver->NR_2ND_STAGE_CONSTRAINTS);

    for (int j=0; j<masterProbSolver->NR_2ND_STAGE_CONSTRAINTS; j++) {
        rhsCoeffs[j] = masterProbSolver->h_s_j[s][j] - myScalarProduct(masterProbSolver->T_s_j_i[s][j], masterProbSolver->masterSol_x_t);
    }
    return rhsCoeffs;
}

void BRP_SubProblem::updateFirstStageVariableValues() {
    // If there are new values for the first-stage decision variables x,
    // we have to update the right-hand sides of some of the constraints in the subproblem

    // Some dimension checking
    int nrConstraints1 = subProbPtr->getOriginalRows();
    int nrConstraints2 = subProbPtr->getRows();
    if (nrConstraints1 != nrConstraints2) throw std::invalid_argument("Disable presolve please, thanks");
    if (nrConstraints1 != masterProbSolver->NR_2ND_STAGE_CONSTRAINTS) throw std::invalid_argument("Disable presolve please, thanks");

    // New right hand side values based on new values of x
    std::vector<double> newRightHandSides = computeNewRightHandSides();

    // Make a vector with integer values from 0 to newRightHandSides.size()-1
    std::vector<int> newRightHandSidesIndices(newRightHandSides.size());
    std::iota(newRightHandSidesIndices.begin(), newRightHandSidesIndices.end(), 0);

    subProbPtr->chgRhs(newRightHandSides.size(), newRightHandSidesIndices, newRightHandSides);
}

void BRP_SubProblem::solveSubProblem() {
    // subProbPtr->writeProb(xpress::format("SubProb_%d.%d.lp", iter, s), "l");

    subProbPtr->optimize();

    // Check the solution status
    if (subProbPtr->getSolStatus() != SolStatus::Optimal && subProbPtr->getSolStatus() != SolStatus::Feasible) {
        std::ostringstream oss; oss << subProbPtr->getSolStatus(); // Convert xpress::SolStatus to String
        throw std::runtime_error("Optimization of subProblem " + std::to_string(s) + " in iteration " +
                                 std::to_string(masterProbSolver->iter) + " failed with status " + oss.str());
    }

    // Optionally print some information
    if (masterProbSolver->verbose) {
        std::cout << "\tScenario " << s << ": Sub Problem Solved" << std::endl;
        std::cout << "\t\tObjective value = " << subProbPtr->getObjVal() << std::endl;
        std::vector<double> solutionValues = subProbPtr->getSolution();

        double nrBikesMovedEndOfDay = 0.0, nrUnmetDemand = 0.0, nrOverflow = 0.0;
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                nrBikesMovedEndOfDay += y[i][j].getValue(solutionValues);
            }
            nrUnmetDemand += u[i].getValue(solutionValues);
            nrOverflow += o[i].getValue(solutionValues);
        }
        std::cout << "\t\tnrBikesMovedEndOfDay = " << nrBikesMovedEndOfDay << std::endl;
        std::cout << "\t\tnrUnmetDemand = " << nrUnmetDemand << std::endl;
        std::cout << "\t\tnrOverflow = " << nrOverflow << std::endl;

        // for (int i=0; i<NR_STATIONS; i++) {
        //     for (int j=0; j<NR_STATIONS; j++) {
        //         std::cout << "\t\t" << y[i][j].getName() << " = " << y[i][j].getValue(solutionValues) << std::endl;
        //     }
        // }
        // for (int i=0; i<NR_STATIONS; i++) {
        //     std::cout << "\t\t" << u[i].getName() << " = " << u[i].getValue(solutionValues) << std::endl;
        // }
        // for (int i=0; i<NR_STATIONS; i++) {
        //     std::cout << "\t\t" << o[i].getName() << " = " << o[i].getValue(solutionValues) << std::endl;
        // }
    }
}



int main() {
    try {
        int nr_stations = 100;
        int nr_scenarios = 50;
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
        std::vector<double> b_i2 = stationData.getColumn<double>("nbDocks");

        // Trip information data:
        std::vector<std::vector<double>> d_s_i2;
        for (const auto& filename : tripDataFilenames) {
            DataFrame tripData = DataFrame::readCSV(filename);
            tripData.convertColumnToDouble("CLASSIC_net");
            std::map<std::string, DataFrame> scenarios = tripData.groupBy<std::string>("date");
            std::vector<std::vector<double>> d_s_i_single = convertScenariosToMatrix(scenarios);
            d_s_i2.insert(d_s_i2.end(), d_s_i_single.begin(), d_s_i_single.end());
        }


        std::vector<double> b_i = b_i2;
        std::vector<std::vector<double>> d_s_i;
        for (int i=0; i<nr_scenarios; i++) {
            d_s_i.push_back(d_s_i2[i]);
        }


        /******************  Data Initialization ******************************/
        // Right-hand coefficients b for each 1st-stage constraint j
        // Note: the nr of 1st-stage constraints == nr first-stage variables, so use index i instead of j
        // std::vector<double> b_i = { 10, 15,  20 };
        // Right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        // Note: the nr of 2nd-stage constraints == nr first-stage variables, so use index i instead of j
        // std::vector<std::vector<double>> d_s_i = {{ 4,  5,  -9}, { -4,  10, -6 }};
        // std::vector<std::vector<double>> d_s_i = {d_s_i2[0]}; // {{ 4,  5,  -9}, {  9,  -3, -6 }};

        int NR_STATIONS = b_i.size();
        int NR_SCENARIOS = d_s_i.size();
        int NR_BIKES = mySum(b_i) / 3 * 2;

        // Objective coefficients c for each first-stage decision variable x_i
        std::vector<double> c_i(NR_STATIONS, 10);
        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        // Objective coefficients for each second-stage variable o_i
        std::vector<double> q_i_1(NR_STATIONS, 10);
        // Objective coefficients for each second-stage variable u_i
        std::vector<double> q_i_2(NR_STATIONS, 10);
        // Probability of each scenario s
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));


        /******************  Problem Creation ******************************/
        // For keeping track of timings and other info
        DataFrame infoDf;
        std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
        // Count duration of everything
        start = std::chrono::high_resolution_clock::now();

        // Create a problem instance
        XpressProblem masterProb;
        // masterProb.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        // Initialize Two-Stage Stochastic Problem solver
        TwoStage_LShapedMethod tssp_solver = 
            TwoStage_LShapedMethod(masterProb, c_i, b_i, p_s, c_ij, q_i_1, q_i_1, d_s_i);


        /******************  Problem Solving ******************************/
        // Solve the TSSP
        tssp_solver.runLShapedMethod(NR_BIKES, false);

        // End of solving
        end = std::chrono::high_resolution_clock::now();
        saveTimeToInfoDf(infoDf, start, end, "Problem Solving (ms)", NR_STATIONS, NR_SCENARIOS);
        // Save number of iterations
        saveDoubleToInfoDf(infoDf, double(tssp_solver.getNumberOfIterations()), "NrIterations", NR_STATIONS, NR_SCENARIOS);
        saveDoubleToInfoDf(infoDf, masterProb.getObjVal(), "ObjectiveVal", NR_STATIONS, NR_SCENARIOS);

        /***************** Showing the Solution **************************/
        std::vector<Variable>& x = tssp_solver.getFirstStageDecisionVariables();
        double theta = tssp_solver.getExpectedSecondStageCost();

        // Print optimal first-stage decisions and objective
        std::cout << std::endl << "*** OPTIMAL SOLUTION FOUND ***" << std::endl;
        std::cout << "Master Objective = " << masterProb.getObjVal() << std::endl;
        std::cout << "E[Q(s,x)] = " << theta << std::endl;
        std::cout << "First Stage Decision Variables:" << std::endl;
        std::vector<double> solution = masterProb.getSolution();
        for (int i=0; i<x.size(); i++) {
            std::cout << "\t" << x[i].getName() << " = " << x[i].getValue(solution) << std::endl;
        }

    }
    catch (std::exception& e) {
        std::cout << "Exception: " << e.what() << std::endl;
        return -1;
    }
}

