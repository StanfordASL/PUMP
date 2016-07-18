DIM ?= 6
NUM ?= 1000
RUNS ?= 1

CXX=nvcc
CXXFLAGS=-O3 -arch=sm_30 -rdc=true -D DIM=$(DIM) -D NUM=$(NUM) -D RUNS=$(RUNS) 

PUMP=testPUMP.cu helper.cu obstacles.cu sampler.cu bvls.cu hsmc.cu collisionCheck.cu 2pBVP.cu CCGMT.cu hardCoded.cu collisionProbability.cu

INFL=2pBVP.cu discreteLQG.cu testInflation.cu CCGMT.cu bvls.cu hardCoded.cu obstacles.cu FMT.cu collisionCheck.cu helper.cu precomp.cu GMT.cu collisionProbability.cu hsmc.cu sampler.cu

pump: $(PUMP)
	$(CXX) $(CXXFLAGS) $(PUMP) -o $@ 

infl: $(INFL)
	$(CXX) $(CXXFLAGS) $(INFL) -o $@ 

clean:
	rm -f *.o *~ *~ pump infl
	rm -rf *.dSYM
