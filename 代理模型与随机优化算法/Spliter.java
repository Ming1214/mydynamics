
/*

*/

import java.util.Random;

public class Spliter {

    private Random rand = new Random();

    public Spliter() {
        super();
    }

    // 
    public SplitData split(double[][] data, double[] target, double pTrain) {
        int N = data.length;
        int dim = data[0].length;
        double[][] x = new double[N][dim];
        double[] y = new double[N];
        int nTrain = 0;
        int nTest = 0;
        for (int i = 0; i < N; i++) {
            if (rand.nextDouble() < pTrain) {
                System.arraycopy(data[i], 0, x[nTrain], 0, dim);
                y[nTrain] = target[i];
                nTrain += 1;
            } else {
                System.arraycopy(data[i], 0, x[N-1-nTest], 0, dim);
                y[N-1-nTest] = target[i];
                nTest += 1;
            }
        }

        SplitData sData = new SplitData();
        sData.trainX = new double[nTrain][dim];
        sData.trainY = new double[nTrain];
        sData.testX = new double[nTest][dim];
        sData.testY = new double[nTest];
        System.arraycopy(x, 0, sData.trainX, 0, nTrain);
        System.arraycopy(y, 0, sData.trainY, 0, nTrain);
        System.arraycopy(x, nTrain, sData.testX, 0, nTest);
        System.arraycopy(y, nTrain, sData.testY, 0, nTest);

        return sData;

    }

}



