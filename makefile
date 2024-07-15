#************************************************
#*  Optimizer: Makefile                         *
#*  =============                               *
#*                                              *
#*  file makefile                               *
#*  `````````````                               *
#*  Makefile for Xpress-Optimzer C++ examples   *
#*                                              *
#*  (c) 2008-2024 Fair Isaac Corporation        *
#*      author: S.Heipcke, 2000                 *
#************************************************

!include makexprb

all: *.cpp
	@$(MAKE) $(**:.cpp=.exe)
# all: *.java
# 	!$(MAKE) /nologo $(**B).class


clean:
	del *.exe
	del *.obj
	del *.mat
	del *.lp
	del *.ilk
	del *.pdb
	del *.class