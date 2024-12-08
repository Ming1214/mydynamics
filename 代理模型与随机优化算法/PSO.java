import java.util.Random;
import java.util.ArrayList;

public class PSO {

    private Random rand = new Random();

    private double xmax = 1.0;
    private double xmin = 0.0;
    private double vmax = 1.0;
    private double vmin = -1.0;

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

    public PSO() {
        super();
    }

    // 
    public void initModel(Assessor ass, int dimension) {
        assessor = ass;
        dim = dimension;
        bestOne = new double[dim];
    }

    // 
    public void optimize(int popSize, double c1, double c2, double w, int maxEpoch, int expectKeep) {
        
        bestValues = new ArrayList();
        meanValues = new ArrayList();
        stdValues = new ArrayList();

        double[][] particles = new double[popSize][dim];
        double[][] speed = new double[popSize][dim];
        for (int i = 0; i < popSize; i++) {
            for (int j = 0; j < dim; j++) {
                particles[i][j] = rand.nextDouble();
                speed[i][j] = 2*rand.nextDouble()-1;
            }
        }
        
        double[] nowValues = assessor.predict(particles);
        double[] st = stat(nowValues);
        bestValues.add(st[0]);
        meanValues.add(st[1]);
        stdValues.add(st[2]);
        for (int i = 0; i < popSize; i++) {
            if (i == 0) {
                System.arraycopy(particles[i], 0, bestOne, 0, dim);
                bestValue = nowValues[i];
            } else {
                if (nowValues[i] > bestValue) {
                    System.arraycopy(particles[i], 0, bestOne, 0, dim);
                    bestValue = nowValues[i];
                }
            }
        }
        keepEpoch = 0;

        for (int e = 0; e < maxEpoch; e++) {
            epoch = e + 1;

            double nowBest = 0.0;
            double[] nowBestOne = new double[dim];
            for (int i = 0; i < popSize; i++) {
                if (i == 0) {
                    nowBest = nowValues[i];
                    System.arraycopy(particles[i], 0, nowBestOne, 0, dim);
                } else {
                    if (nowValues[i] > nowBest) {
                        nowBest = nowValues[i];
                        System.arraycopy(particles[i], 0, nowBestOne, 0, dim);
                    }
                }
            }

            for (int i = 0; i < popSize; i++) {
                for (int j = 0; j < dim; j++) {
                    speed[i][j] = w*speed[i][j] + c1*rand.nextDouble()*(nowBestOne[j]-particles[i][j]) + c2*rand.nextDouble()*(bestOne[j]-particles[i][j]);
                    particles[i][j] += speed[i][j];
                    if (speed[i][j] < vmin) {
                        speed[i][j] = vmin;
                    }
                    if (speed[i][j] > vmax) {
                        speed[i][j] = vmax;
                    }
                    if (particles[i][j] > xmax) {
                        particles[i][j] = xmax;
                    }
                    if (particles[i][j] < xmin) {
                        particles[i][j] = xmin;
                    }
                }
            }

            nowValues = assessor.predict(particles);
            st = stat(nowValues);
            bestValues.add(st[0]);
            meanValues.add(st[1]);
            stdValues.add(st[2]);
            boolean notChanged = true;
            for (int i = 0; i < popSize; i++) {
                if (nowValues[i] > bestValue) {
                    bestValue = nowValues[i];
                    System.arraycopy(particles[i], 0, bestOne, 0, dim);
                    notChanged = false;
                }
            }
            if (notChanged) {
                keepEpoch += 1;
                if (keepEpoch >= expectKeep) {
                    break;
                }
            } else {
                keepEpoch = 0;
            }
        }

    }

    // 
    private double[] stat(double[] values) {
        int N = values.length;
        double best = values[0];
        double mean = 0.0;
        double std = 0.0;
        for (int i = 0; i < N; i++) {
            if (values[i] > best) {
                best = values[i];
            }
            mean += values[i] / N;
        }
        for (int i = 0; i < N; i++) {
            std += Math.pow(values[i]-mean, 2) / (N - 1);
        }
        std = Math.sqrt(std);
        double[] st = {best, mean, std};
        return st;
    }

}



