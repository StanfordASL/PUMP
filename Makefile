DIM ?= 6
NUM ?= 1000

CXX=nvcc
CXXFLAGS=-O3 -arch=sm_52 -rdc=true -D DIM=$(DIM) -D NUM=$(NUM)

PUMP=mainPUMP.cu helper.cu obstacles.cu sampler.cu bvls.cu hsmc.cu collisionCheck.cu 2pBVP.cu PUMP.cu hardCoded.cu collisionProbability.cu motionPlanningProblem.cu roadMap.cu

pump: $({PUMP})
	$(CXX) $(CXXFLAGS) $(PUMP) -o $@ 

clean:
	rm -f *.o *~ *~ pump
	rm -rf *.dSYM
