import java.util.Random;
import java.util.ArrayList;

public class GA {

    private Random rand = new Random();

    private Assessor assessor;
    private int dim;

    // 
    public double[] bestOne;
    public double bestValue;
    public int keepEpoch;
    public int epoch;

    // 
    public ArrayList bestValues;
    public ArrayList meanValues;
    public ArrayList stdValues;

    public GA() {
        super();
    }

    // 
    public void initModel(Assessor ass, int dimension) {
        assessor = ass;
        dim = dimension;
        bestOne = new double[dim];
    }

    // 
    public void optimize(int popSize, double pCrossover, double pMutation, int maxEpoch, int expectKeep) {
        bestValues = new ArrayList();
        meanValues = new ArrayList();
        stdValues = new ArrayList();
        double[][] population = new double[popSize][dim];
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < dim; j++) {
                population[i][j] = rand.nextDouble();
            }
        }
        sort(population);
        double[] st = stat(population);
        bestValues.add(st[0]);
        meanValues.add(st[1]);
        stdValues.add(st[2]);
        System.arraycopy(population[0], 0, bestOne, 0, dim);
        bestValue = st[0];
        keepEpoch = 0;
        for (int e = 0; e < maxEpoch; e++) {
            epoch = e + 1;
            double[][] nextGeneration = new double[2*popSize][dim];
            nextGeneration = crossOver(population, pCrossover);
            nextGeneration = mutation(nextGeneration, pMutation);
            population = select(population, nextGeneration);
            st = stat(population);
            bestValues.add(st[0]);
            meanValues.add(st[1]);
            stdValues.add(st[2]);
            if (st[0] > bestValue) {
                System.arraycopy(population[0], 0, bestOne, 0, dim);
                bestValue = st[0];
                keepEpoch = 0;
            } else {
                keepEpoch += 1;
                if (keepEpoch >= expectKeep) {
                    break;
                }
            }
        }
    }

    // 
    private double[][] crossOver(double[][] pop, double p) {
        int N = pop.length;
        double[][] newPop = new double[2*N][dim];
        for (int i = 0; i < N; i++) {
            int j = 0;
            for (; j < N; j++) {
                if (j == i) {
                    continue;
                }
                if (rand.nextDouble() < 0.5) {
                    break;
                }
            }
            if (j == N) {
                j = 0;
            }
            System.arraycopy(pop[i], 0, newPop[2*i], 0, dim);
            System.arraycopy(pop[j], 0, newPop[2*i+1], 0, dim);
            for (int k = 0; k < dim; k++) {
                if (rand.nextDouble() < p) {
                    newPop[2*i][k] = pop[j][k];
                    newPop[2*i+1][k] = pop[i][k];
                }
            }
        }
        return newPop;
    }

    // 
    private double[][] mutation(double[][] pop, double p) {
        int N = pop.length;
        double[][] newPop = new double[N][dim];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dim; j++) {
                if (rand.nextDouble() < p) {
                    newPop[i][j] = rand.nextDouble();
                } else {
                    newPop[i][j] = pop[i][j];
                }
            }
        }
        return newPop;
    }

    // 
    private double[][] select(double[][] pop1, double[][] pop2) {
        int N1 = pop1.length;
        int N2 = pop2.length;
        double[][] newPop = new double[N1][dim];
        double[][] pop = new double[N1+N2][dim];
        System.arraycopy(pop1, 0, pop, 0, N1);
        System.arraycopy(pop2, 0, pop, N1, N2);
        sort(pop);
        System.arraycopy(pop, 0, newPop, 0, N1);
        return newPop;
    }

    // 
    private void sort(double[][] pop) {
        int N = pop.length;
        double[] values = assessor.predict(pop);
        for (int i = 0; i < N-1; i++) {
            for (int j = 0; j < N-1-i; j++) {
                if (values[j] < values[j+1]) {
                    double tmpV = values[j];
                    values[j] = values[j+1];
                    values[j+1] = tmpV;
                    double[] tmpP = new double[dim];
                    System.arraycopy(pop[j], 0, tmpP, 0, dim);
                    System.arraycopy(pop[j+1], 0, pop[j], 0, dim);
                    System.arraycopy(tmpP, 0, pop[j+1], 0, dim);
                }
            }
        }
    }

    // 
    private double[] stat(double[][] pop) {
        int N = pop.length;
        double[] values = assessor.predict(pop);
        double best = values[0];
        double mean = 0.0;
        double std = 0.0;
        for (int i = 0; i < N; i++) {
            mean += values[i] / N;
        }
        for (int i = 0; i < N; i++) {
            std += Math.pow(values[i]-mean, 2) / (N-1);
        }
        std = Math.sqrt(std);
        double[] st = {best, mean, std};
        return st;
    }

}



