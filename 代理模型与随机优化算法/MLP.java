import java.util.Random;
import java.util.ArrayList;

public class MLP extends Assessor {
    
    private Random rand = new Random();

    // 
    private double[][] weightHidden;
    private double[] biasHidden;
    private double[] weightOutput;
    private double biasOutput;
 
    // 
    private double[][] gradWeightHidden;
    private double[] gradBiasHidden;
    private double[] gradWeightOutput;
    private double gradBiasOutput;

    // 
    private double[] input;
    private double[] preHidden;
    private double[] hidden;
    private double preOutput;
    private double output;
    private double target;

    // 
    private int inputNode;
    private int hiddenNode;
    
    // 
    private String hiddenFunc;
    private String outputFunc;

    // 
    public int epoch;
    public double mseTrain;
    public double r2Train;
    public double mseTest;
    public double r2Test;

    // 
    public ArrayList trainMse;
    public ArrayList trainR2;
    public ArrayList testMse;
    public ArrayList testR2;

    public MLP() {
        super();
    }

    // 
    public void initModel(int dInputNode, int dHiddenNode, String dHiddenFunc, String dOutputFunc) {
        inputNode = dInputNode;
        hiddenNode = dHiddenNode;
        hiddenFunc = dHiddenFunc;
        outputFunc = dOutputFunc;
        weightHidden = new double[hiddenNode][inputNode];
        biasHidden = new double[hiddenNode];
        weightOutput = new double[hiddenNode];
        gradWeightHidden = new double[hiddenNode][inputNode];
        gradBiasHidden = new double[hiddenNode];
        gradWeightOutput = new double[hiddenNode];
        input = new double[inputNode];
        preHidden = new double[hiddenNode];
        hidden = new double[hiddenNode];
    }

    // 
    public void initParams() {
        for (int i = 0; i < hiddenNode; i++) {
            for (int j = 0; j < inputNode; j++) {
                weightHidden[i][j] = rand.nextGaussian();
            }
            biasHidden[i] = rand.nextGaussian();
            weightOutput[i] = rand.nextGaussian();
        }
        biasOutput = rand.nextGaussian();
    }

    // 
    private double actFunction(double x, String func) {
        if (func == "linear") {
            return x;
        }
        if (func == "sigmoid") {
            return 1.0 / (1.0 + Math.exp(-x));
        }
        if (func == "tanh") {
            return Math.tanh(x);
        }
        if (func == "relu") {
            if (x > 0) {
                return x;
            } else {
                return 0;
            }
        }
        return x;
    }

    // 
    private double gradActFunction(double x, String func) {
        if (func == "linear") {
            return 1.0;
        }
        if (func == "sigmoid") {
            double y = actFunction(x, func);
            return y * (1-y);
        }
        if (func == "tanh") {
            double y = actFunction(x, func);
            return 1 - Math.pow(y, 2);
        }
        if (func == "relu") {
            if (x > 0) {
                return 1.0;
            } else {
                return 0.0;
            }
        }
        return 1.0;
    }

    // 
    private void forward(double[] dInput) {
        input = dInput;
        preOutput = 0.0;
        for (int i = 0; i < hiddenNode; i++) {
            preHidden[i] = 0.0;
            for (int j = 0; j < inputNode; j++) {
                preHidden[i] += weightHidden[i][j] * input[j];
            }
            preHidden[i] += biasHidden[i];
            hidden[i] = actFunction(preHidden[i], hiddenFunc);
            preOutput += weightOutput[i] * hidden[i];
        }
        preOutput += biasOutput;
        output = actFunction(preOutput, outputFunc);
    }

    // 
    private void backpropagation(double dTarget) {
        target = dTarget;
        gradBiasOutput = (output-target) * gradActFunction(preOutput, outputFunc);
        for (int i = 0; i < hiddenNode; i++) {
            gradWeightOutput[i] = gradBiasOutput * hidden[i];
            gradBiasHidden[i] = gradBiasOutput * weightOutput[i] * gradActFunction(preHidden[i], hiddenFunc);
            for (int j = 0; j < inputNode; j++) {
                gradWeightHidden[i][j] = gradBiasHidden[i] * input[j];
            }
        }
    }

    // 
    private void trainOneBatch(double[][] dInputs, double[] dTargets, double learning_rate) {
        int N = dInputs.length;
        double[][] batchGradWeightHidden = new double[hiddenNode][inputNode];
        double[] batchGradBiasHidden = new double[hiddenNode];
        double[] batchGradWeightOutput = new double[hiddenNode];
        double batchGradBiasOutput = 0.0;
        for (int n = 0; n < N; n++) {
            forward(dInputs[n]);
            backpropagation(dTargets[n]);
            batchGradBiasOutput += gradBiasOutput / N;
            for (int i = 0; i < hiddenNode; i++) {
                batchGradWeightOutput[i] += gradWeightOutput[i] / N;
                batchGradBiasHidden[i] += gradBiasHidden[i] / N;
                for (int j = 0; j < inputNode; j++) {
                    batchGradWeightHidden[i][j] += gradWeightHidden[i][j] / N;
                }
            }
        }
        biasOutput -= learning_rate * batchGradBiasOutput;
        for (int i = 0; i < hiddenNode; i++) {
            weightOutput[i] -= learning_rate * batchGradWeightOutput[i];
            biasHidden[i] -= learning_rate * batchGradBiasHidden[i];
            for (int j = 0; j < inputNode; j++) {
                weightHidden[i][j] -= learning_rate * batchGradWeightHidden[i][j];
            }
        }
    }

    // 
    @Override
    public double predictOnePiece(double[] dInput) {
        double o = 0.0;
        for (int i = 0; i < hiddenNode; i++) {
            double h_i = 0.0;
            for (int j = 0; j < inputNode; j++) {
                h_i += dInput[j] * weightHidden[i][j];
            }
            h_i += biasHidden[i];
            h_i = actFunction(h_i, hiddenFunc);
            o += h_i * weightOutput[i];
        }
        o += biasOutput;
        o = actFunction(o, outputFunc);
        return o;
    }

    // 
    @Override
    public double[] predict(double[][] dInputs) {
        int N = dInputs.length;
        double[] os = new double[N];
        for (int n = 0; n < N; n++) {
            os[n] = predictOnePiece(dInputs[n]);
        }
        return os;
    }

    // 
    public double mse(double[][] dInputs, double[] dTargets) {
        int N = dInputs.length;
        double[] os = predict(dInputs);
        double mse = 0.0;
        for (int n = 0; n < N; n++) {
            mse += Math.pow(os[n]-dTargets[n], 2) / N;
        }
        return mse;
    }

    // 
    public double r2(double[][] dInputs, double[] dTargets) {
        int N = dInputs.length;
        double[] os = predict(dInputs);
        double meanOs = 0.0;
        double meanTs = 0.0;
        for (int n = 0; n < N; n++) {
            meanOs += os[n]/N;
            meanTs += dTargets[n]/N;
        }
        double s1 = 0.0;
        double s2 = 0.0;
        double s12 = 0.0;
        for (int n = 0; n < N; n++) {
            s1 += Math.pow(os[n]-meanOs, 2);
            s2 += Math.pow(dTargets[n]-meanTs, 2);
            s12 += (os[n]-meanOs)*(dTargets[n]-meanTs);
        }
        return Math.pow(s12, 2) / (s1 * s2);
    }

    // 
    public void train(double[][] trainInputs, double[] trainTargets, double[][] testInputs, double[] testTargets, int maxEpoch, int batchSize, double expectR2, double learning_rate) {
            int N = trainInputs.length;
            trainMse = new ArrayList();
            trainR2 = new ArrayList();
            testMse = new ArrayList();
            testR2 = new ArrayList();
            for (int e = 0; e < maxEpoch; e++) {
                epoch = e + 1;
                for (int eb = 0; eb*batchSize < N; eb++) {
                    int size = batchSize;
                    if ((eb+1)*batchSize >= N) {
                        size = N - eb*batchSize;
                    }
                    double[][] batchInputs = new double[size][inputNode];
                    double[] batchTargets = new double[size];
                    System.arraycopy(trainInputs, eb*batchSize, batchInputs, 0, size);
                    System.arraycopy(trainTargets, eb*batchSize, batchTargets, 0, size);
                    trainOneBatch(batchInputs, batchTargets, learning_rate);
                }
                mseTrain = mse(trainInputs, trainTargets);
                r2Train = r2(trainInputs, trainTargets);
                mseTest = mse(testInputs, testTargets);
                r2Test = r2(testInputs, testTargets);
                trainMse.add(mseTrain);
                trainR2.add(r2Train);
                testMse.add(mseTest);
                testR2.add(r2Test);
                if (r2Train >= expectR2 && r2Test >= expectR2) {
                    break;
                }
            }
        }

}



