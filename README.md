This repository contains 6 main files, all solving the same problem in a different way.
The problem that is solved is an instance of a Two-Stage Stochastic Problem (TSSP).
TSSP's can be solved in different ways. The book `Introduction to Stochastic Programming` 
by Birge and Louveaux can provide further context to those unfamiliar with TSSP.

There are two different formulations for the same problem:
    1. Main formulation. (Using variables `u`)
    2. Alternative formulation. (Using variables `u` and `o`)

Each formulation is solved in 3 ways: 
    1. Deterministic Equivalent Problem
    2. L-Shaped Method
    3. Enhanced L-Shaped Method

This gives rise to the 3*2=6 main files, all solving the same problem in a different way.
Section 5.1.a in the mentioned book provides an excellent explanation of the L-shaped method 
and optimality cuts as used in the code in this folder. Also, documentation in the comments
of the 6 main files further explain which of the two formulations is used, and which of the
3 solution methods is applied, and how.

Furthermore, there are two helper files:
    1. DataFrame.h, for common data operations comparable to Pandas in Python
    2. BrpUtils.h, which contains some functions that all 6 of the problem-files call.
            These include data-reading and data-writing operations, numerical operations, etc.

Finally, there are the Makefiles to compile and execute the .cpp code