#include <xpress.hpp>
#include <stdexcept> // For throwing exceptions
#include <unordered_map>
#include <chrono>
#include "DataFrame.h"

using namespace xpress;
using namespace xpress::objects;

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


class TwoStage_LShapedMethod {
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
    std::vector<std::vector<double>> q_ij_1, q_ij_2;
    // Right hand coefficients h for some 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<std::vector<double>>> d_s_ij;

    // Constructor method: give all required coefficients / data
    TwoStage_LShapedMethod(XpressProblem& masterProb, std::vector<double>& c_i, std::vector<double>& b_i, 
        std::vector<double>& p_s, std::vector<std::vector<double>>& c_ij, std::vector<std::vector<double>>& q_ij_1,
        std::vector<std::vector<double>>& q_ij_2, std::vector<std::vector<std::vector<double>>>& d_s_ij);

    void runLShapedMethod(int NR_BIKES, bool verbose);
    std::vector<Variable>& getFirstStageDecisionVariables();
    double getExpectedSecondStageCost();


private:
    // First-stage decision variables
    std::vector<Variable> x;
    // Auxiliary variable in the master problem - necessary for the decomposition
    Variable theta;

    void makeInitialMasterProbFormulation(int NR_BIKES);
    void solveMasterProb();
    void addOptimalityCutToMasterProb(std::vector<double>& E_t, double& e_t);
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

    std::vector<std::vector<double>> netDemand_s_i;
};

// Constructor Method
TwoStage_LShapedMethod::TwoStage_LShapedMethod(XpressProblem& masterProb,
        std::vector<double>& c_i, std::vector<double>& b_i, std::vector<double>& p_s,
        std::vector<std::vector<double>>& c_ij, std::vector<std::vector<double>>& q_ij_1,
        std::vector<std::vector<double>>& q_ij_2, std::vector<std::vector<std::vector<double>>>& d_s_ij) 
     : masterProb(masterProb), c_i(c_i), b_i(b_i), p_s(p_s),
       c_ij(c_ij), q_ij_1(q_ij_1), q_ij_2(q_ij_2), d_s_ij(d_s_ij)
    {
    this->iter                      = 0;
    this->NR_SCENARIOS              = p_s.size();
    this->NR_1ST_STAGE_VARIABLES    = c_i.size();
    this->NR_1ST_STAGE_CONSTRAINTS  = b_i.size();
    NR_STATIONS = NR_1ST_STAGE_VARIABLES;

    this->NR_2ND_STAGE_VARIABLES    = 2 * NR_STATIONS * NR_STATIONS;
    this->NR_2ND_STAGE_CONSTRAINTS  = 3 * NR_STATIONS;

    this->netDemand_s_i = std::vector<std::vector<double>>(NR_SCENARIOS, std::vector<double>(NR_STATIONS, 0));
    for (int s=0; s<NR_SCENARIOS; s++) {
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                netDemand_s_i[s][i] += d_s_ij[s][i][j];
                netDemand_s_i[s][j] -= d_s_ij[s][i][j];
            }
        }
    }
    for (int s=0; s<NR_SCENARIOS; s++) {
        std::cout << "SCENARIO " << s << std::endl;
        for (int i=0; i<NR_STATIONS; i++) {
            std::cout << "d[" << i << "] = " << netDemand_s_i[s][i] << std::endl;
        }
        std::cout << std::endl;
    }
}

void TwoStage_LShapedMethod::runLShapedMethod(int NR_BIKES, bool verbose) {

    this->verbose = verbose;
    makeInitialMasterProbFormulation(NR_BIKES);

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

double TwoStage_LShapedMethod::getExpectedSecondStageCost() {
    return masterProb.getSolution(theta);
}

void TwoStage_LShapedMethod::makeInitialMasterProbFormulation(int NR_BIKES) {
    /* VARIABLES */
    this->x = masterProb.addVariables(NR_1ST_STAGE_VARIABLES).withType(ColumnType::Integer)
            .withName([](int i){ return xpress::format("x_%d", i); }).toArray();

    /* CONSTRAINTS */
    masterProb.addConstraints(NR_1ST_STAGE_VARIABLES, [&](int i) {
        return (x[i] <= b_i[i]).setName(xpress::format("Capacity%d", i));
    });
    masterProb.addConstraint(Utils::sum(x) == NR_BIKES).setName("Nr bikes constraint");


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
    std::cout << "\tMaster Problem Solved" << std::endl;
    if (verbose) {
        std::cout << "\t\tMaster Objective = " << masterProb.getObjVal() << std::endl;
        for (int i=0; i<x.size(); i++) std::cout << "\t\t" << x[i].getName() << " = " << masterSol_x_t[i] << std::endl;
        std::cout << "\t\ttheta = " << (masterSol_theta_t == XPRS_MINUSINFINITY ? "MINUS INFINITY" : std::to_string(masterSol_theta_t)) << std::endl;
        std::cout << std::endl;
    }
}

void TwoStage_LShapedMethod::addOptimalityCutToMasterProb(std::vector<double>& E_t, double& e_t) {
    masterProb.addConstraint(Utils::scalarProduct(x, E_t) + theta >= e_t);
    std::cout << "\tAdding constraint: " << (Utils::scalarProduct(x, E_t) + theta).toString() << " >= " << e_t << std::endl << std::endl;
}

bool TwoStage_LShapedMethod::generateOptimalityCut(std::vector<double>& E_t, double& e_t) {

    // ################## Solving Sub Problems ######################
    
    // To store the right hand coefficients h for each 2nd-stage constraint j, for each scenario s
    std::vector<std::vector<double>> h_s_j(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));
    // To store the constraint coefficients T for each 1st-stage variable x_i, for each 2nd-stage constraints j, for each scenario s
    // std::vector<std::vector<std::vector<double>>> T_s_i_j(NR_SCENARIOS, std::vector<std::vector<double>>(NR_1ST_STAGE_VARIABLES, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS, 0.0)));
    std::vector<std::vector<std::vector<double>>> T_s_j_i(NR_SCENARIOS, std::vector<std::vector<double>>(NR_2ND_STAGE_CONSTRAINTS, std::vector<double>(NR_1ST_STAGE_VARIABLES, 0.0)));

    // To store the dual values pi for each 2nd-stage constraints j, for each scenario s
    std::vector<std::vector<double>> pi_s_j(NR_SCENARIOS, std::vector<double>(NR_2ND_STAGE_CONSTRAINTS));

    for (int s=0; s<NR_SCENARIOS; s++) {
        XpressProblem subProb_s;
        /* VARIABLES */
        std::vector<std::vector<Variable>> y = subProb_s.addVariables(NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Continuous)
            // .withName([s](int i, int j){ return xpress::format("s%d_y(%d,%d)", s, i, j); })
            .toArray();

        std::vector<std::vector<Variable>> u = subProb_s.addVariables(NR_STATIONS, NR_STATIONS)
            .withType(ColumnType::Continuous)
            // .withName([s](int i, int j){ return xpress::format("s%d_u(%d,%d)", s, i, j); })
            .toArray();
        
        std::cout << "\tBuilt sub problem variables" << std::endl;

        /* CONSTRAINTS */

        for (int i=0; i<NR_STATIONS; i++) {
            h_s_j[s][0*NR_STATIONS+i]      = netDemand_s_i[s][i];
            h_s_j[s][1*NR_STATIONS+i]      = b_i[i] - netDemand_s_i[s][i];
            h_s_j[s][2*NR_STATIONS+i]      = - netDemand_s_i[s][i];

            T_s_j_i[s][0*NR_STATIONS+i][i] = 0.0;
            T_s_j_i[s][1*NR_STATIONS+i][i] = 1.0;
            T_s_j_i[s][2*NR_STATIONS+i][i] = 1.0;
        }

        std::vector<LinExpression> end_of_day_net_recourse_flows(NR_STATIONS);
        std::vector<Expression> during_day_net_customer_flows(NR_STATIONS);
        std::vector<LinExpression> nr_disabled_incoming_trips(NR_STATIONS);
        std::vector<LinExpression> nr_disabled_outgoing_trips(NR_STATIONS);

        for (int i=0; i<NR_STATIONS; i++) {
            LinExpression net_recourse_flow = LinExpression::create();
            LinExpression outgoing_trips = LinExpression::create();
            LinExpression incoming_trips = LinExpression::create();
            for (int j=0; j<NR_STATIONS; j++) {
                net_recourse_flow.addTerm(y[i][j], 1).addTerm(y[j][i], -1);
                outgoing_trips.addTerm(u[i][j], 1);
                incoming_trips.addTerm(u[j][i], 1);
            }
            end_of_day_net_recourse_flows[i] = net_recourse_flow;
            nr_disabled_outgoing_trips[i] = outgoing_trips;
            nr_disabled_incoming_trips[i] = incoming_trips;
            during_day_net_customer_flows[i] = -(netDemand_s_i[s][i] - 
                nr_disabled_outgoing_trips[i] + nr_disabled_incoming_trips[i]
            );
        }

        subProb_s.addConstraints(NR_STATIONS, [&](int i) {
            return (end_of_day_net_recourse_flows[i] == during_day_net_customer_flows[i])
                    .setName(xpress::format("FlowCons_S%d", i));
        });
        subProb_s.addConstraints(NR_STATIONS, [&](int i) {
            return (masterSol_x_t[i] + during_day_net_customer_flows[i] <= b_i[i]);
                    // .setName(xpress::format("FlowCons_S%d", i));
        });
        subProb_s.addConstraints(NR_STATIONS, [&](int i) {
            return (masterSol_x_t[i] + during_day_net_customer_flows[i] >= 0.0);
                    // .setName(xpress::format("FlowCons_S%d", i));
        });

        std::cout << "\tBuilt sub problem constraints" << std::endl;

        /* OBJECTIVE */
        LinExpression objective = LinExpression::create();
        for (int i=0; i<NR_STATIONS; i++) {
            for (int j=0; j<NR_STATIONS; j++) {
                objective.addTerm(c_ij[i][j], y[i][j]);
                objective.addTerm(q_ij_1[i][j], u[i][j]);
            }
        }
        // objective.addConstant(myScalarProduct(u_i, q_i_1) + myScalarProduct(o_i, q_i_2));
        subProb_s.setObjective(objective, xpress::ObjSense::Minimize);

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
        std::vector<double> solutionValues = subProb_s.getSolution();
        // TODO: improve getSolution() double for loop
        // for (int i=0; i<NR_STATIONS; i++) {
        //     for (int j=0; j<NR_STATIONS; j++) {
        //         std::cout << "\t\t" << y[i][j].getName() << " = " << y[i][j].getSolution() << std::endl;
        //     }
        // }

        if (verbose) {
            for (int i=0; i<NR_STATIONS; i++) {
                std::cout << "\t\tStation " << i << std::endl;
                std::cout << "\t\t\tend-of-day-flow = " << -end_of_day_net_recourse_flows[i].evaluate(solutionValues) << std::endl;
                std::cout << "\t\t\tunmet-outgoing-trips = " << nr_disabled_outgoing_trips[i].evaluate(solutionValues) << std::endl;
                std::cout << "\t\t\tunmet-incoming-trips = " << nr_disabled_incoming_trips[i].evaluate(solutionValues) << std::endl;
                std::cout << "\t\t\tduring-day-flow = " << during_day_net_customer_flows[i].evaluate(solutionValues) << std::endl;
            }
        }

        pi_s_j[s] = subProb_s.getDuals();
        if (pi_s_j[s].size() != NR_2ND_STAGE_CONSTRAINTS) throw std::invalid_argument("Disable presolve please, thanks");

        if (verbose) {
            std::cout << "\t\tpi_s" << s << " = ";
            std::vector<double> duals = subProb_s.getDuals();
            for (int i=0; i<duals.size(); i++) std::cout << duals[i] << ",  ";
            std::cout << std::endl << std::endl;
        }
    }

    for (int s=0; s<NR_SCENARIOS; s++) {
        e_t += p_s[s] * myScalarProduct(pi_s_j[s], h_s_j[s]);
        std::vector<double> result = myMultiplyMatrices(std::vector<std::vector<double>>{pi_s_j[s]}, T_s_j_i[s])[0];
        // E_t = myElementWiseAddition(E_t, myElementWiseMultiplication(p_s[s], result));
        for (int i=0 ; i<NR_1ST_STAGE_VARIABLES; i++) {
            E_t[i] += p_s[s] * result[i];
        }
    }
    std::cout << "\tGenerated Cut:" << std::endl;
    std::cout << "\t\te_" << iter << " = " << e_t << std::endl;
    std::cout << "\t\tE_" << iter << " = ";
    for (int i=0; i<NR_1ST_STAGE_VARIABLES; i++) std::cout << E_t[i] << ",  ";
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
        std::string stationDataFilename = "./Station_Info.csv";
        DataFrame stationData = DataFrame::readCSV(stationDataFilename);
        stationData.convertColumnToDouble("nbDocks");

        // std::string tripDataFilename = "./394_Net_Data.csv";
        // DataFrame tripData    = DataFrame::readCSV(tripDataFilename);
        // tripData.convertColumnToDouble("CLASSIC_net");
        // stationData.convertColumnToDouble("nbDocks");
        // std::map<std::string, DataFrame> scenarios = tripData.groupBy<std::string>("date");
        
        std::string tripDataFilename = "./394_matrix_data_2024_04_30.csv";
        DataFrame tripData    = DataFrame::readCSV(tripDataFilename);
        for (std::string colName : tripData.columnNames()) {
            tripData.convertColumnToDouble(colName);
        }
        std::vector<DataFrame> scenarioData = {tripData};
        int NR_SCENARIOS = scenarioData.size();
        int NR_STATIONS = tripData.length();
        int NR_BIKES = 3000;

        std::vector<std::vector<std::vector<double>>> d_s_ij2(NR_SCENARIOS, std::vector<std::vector<double>>(NR_STATIONS, std::vector<double>(NR_STATIONS)));
        for (int s=0; s<NR_SCENARIOS; s++) {
            for (std::string colName : scenarioData[s].columnNames()) {
                int colIndex = std::stoi(colName);
                std::vector<double> colValues = tripData.getColumn<double>(colName);
                for (int i=0; i<NR_STATIONS; i++) {
                    d_s_ij2[s][i][colIndex] = colValues[i];
                }
            }
        }

        std::vector<double> b_i2 = stationData.getColumn<double>("nbDocks");
        

        /******************  Data Initialization ******************************/
        // Objective coefficients c for each first-stage decision variable x_i
        // int NR_STATIONS = 3;
        std::vector<double> c_i(NR_STATIONS, 10);
        // Right-hand coefficients b for each 1st-stage constraint j
        // Note: the nr of 1st-stage constraints == nr first-stage variables, so use index i instead of j
        std::vector<double> b_i = b_i2;//{ 10, 15,  20 };

        // Right hand coefficients h for each 2nd-stage constraint j, for each scenario s
        // Note: the nr of 2nd-stage constraints == nr first-stage variables, so use index i instead of j
        // std::vector<std::vector<double>> d_s_i = {{ 4,  -5,  9,  -8 }, { 6,  -4,  10,  -7 }};
        std::vector<std::vector<std::vector<double>>> d_s_ij = d_s_ij2; //{{{ 0,   3,   2 }, { 6,  0,  10 }, { 3,  2, 0}},
                                                                         //{{ 0,   5,   6 }, { 2,  0,   2 }, { 7,  2, 0}}};

        // Objective coefficients for each second-stage decision variable y_ij
        std::vector<std::vector<double>> c_ij(NR_STATIONS, std::vector<double>(NR_STATIONS, 1));
        // Objective coefficients for each second-stage constant o_ij
        std::vector<std::vector<double>> q_ij_1(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));
        // Objective coefficients for each second-stage constant u_ij
        std::vector<std::vector<double>> q_ij_2(NR_STATIONS, std::vector<double>(NR_STATIONS, 10));

        // Probability of each scenario s
        // int NR_SCENARIOS = d_s_ij.size();
        std::vector<double> p_s(NR_SCENARIOS, 1/double(NR_SCENARIOS));


        /******************  Problem Creation ******************************/
        // Create a problem instance
        XpressProblem masterProb;
        // masterProb.callbacks->addMessageCallback(XpressProblem::CallbackAPI::console);

        // Initialize Two-Stage Stochastic Problem solver
        TwoStage_LShapedMethod tssp_solver = 
            TwoStage_LShapedMethod(masterProb, c_i, b_i, p_s, c_ij, q_ij_1, q_ij_1, d_s_ij);


        /******************  Problem Solving ******************************/
        // Solve the TSSP
        tssp_solver.runLShapedMethod(NR_BIKES, false);
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

