#include "motionPlanningProblem.cuh"

void printMotionPlanningProblem(MotionPlanningProblem mpp, std::ostream& stream) {
	stream << "--- Motion planning problem, " << mpp.filename << " ---" << std::endl;
	stream << "Sample count = " << mpp.roadmap.numSamples << ", C-space dim = " << mpp.dimX << ", Workspace dim = " << mpp.dimW << std::endl;
	stream << "hi = ["; for (int i = 0; i < mpp.dimX; ++i) { stream << mpp.hi[i] << " "; } stream << "], ";
	stream << "lo = ["; for (int i = 0; i < mpp.dimX; ++i) { stream << mpp.lo[i] << " "; } stream << "]" << std::endl;
	stream << "edge discretizations " << mpp.numDisc << ", dt = " << mpp.dt << std::endl;
	stream << "the " << mpp.numObstacles << " obstacles are = " << std::endl;
	printArray(mpp.obstacles.data(), mpp.numObstacles, 2*DIM, stream);
	stream << "--- PUMP specific" << std::endl;
	stream << "numHSMCParticles = " << mpp.numHSMCParticles << std::endl;
}