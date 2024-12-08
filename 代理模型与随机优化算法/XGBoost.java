import java.util.ArrayList;
import java.util.Iterator;


public class XGBoost extends Assessor {

    // 
    private ArrayList trees;
    private double shrinkRatio;

    // 
    private int dim;
    private double[] vMin;
    private double[] vMax;

    //
    public int treeNum;
    public double mseTrain;
    public double r2Train;
    public double mseTest;
    public double r2Test;

    //
    public ArrayList trainMse;
    public ArrayList trainR2;
    public ArrayList testMse;
    public ArrayList testR2;
    public ArrayList depths;

    public XGBoost() {
        super();
    }

    //
    public void initModel(int dDim, double[] dMin, double[] dMax) {
        dim = dDim;
        vMin = dMin;
        vMax = dMax;
    }

    //
    public void train(double[][] trainX, double[] trainY, double[][] testX, double[] testY, 
                        int tree_num, double shrink, int max_depth, int min_split_num, double min_gain, int interval_num, double expectR2, double lamda) {
        trees = new ArrayList();
        shrinkRatio = shrink;
        treeNum = 0;
        trainMse = new ArrayList();
        trainR2 = new ArrayList();
        testMse = new ArrayList();
        testR2 = new ArrayList();
        depths = new ArrayList();
        double[] trainG = new double[trainY.length];
        System.arraycopy(trainY, 0, trainG, 0, trainY.length);
        for (int i = 0; i < tree_num; i++) {
            XGBCart cart = new XGBCart();
            cart.initModel(dim, vMin, vMax);
            cart.train(trainX, trainG, max_depth, min_split_num, min_gain, interval_num, lamda);
            if (cart.depth == 0) {
                break;
            }
            treeNum = i + 1;
            depths.add(cart.depth);
            trees.add(cart);
            mseTrain = mse(trainX, trainY);
            r2Train = r2(trainX, trainY);
            mseTest = mse(testX, testY);
            r2Test = r2(testX, testY);
            trainMse.add(mseTrain);
            trainR2.add(r2Train);
            testMse.add(mseTest);
            testR2.add(r2Test);
            if (r2Train >= expectR2 && r2Test >= expectR2) {
                break;
            }
            for (int j = 0; j < trainG.length; j++) {
                trainG[j] -= Math.pow(shrink, i+1)*cart.predictOnePiece(trainX[j]);
            }
        }
    }

    //
    @Override
    public double predictOnePiece(double[] X) {
        double y = 0.;
        Iterator iter = trees.iterator();
        int i = 0;
        while (iter.hasNext()) {
            i++;
            XGBCart cart = (XGBCart) iter.next();
            y += Math.pow(shrinkRatio, i)*cart.predictOnePiece(X);
        }
        return y;
    }

    //
    @Override
    public double[] predict(double[][] X) {
        int N = X.length;
        double[] os = new double[N];
        for (int n = 0; n < N; n++) {
            os[n] = predictOnePiece(X[n]);
        }
        return os;
    }
 
    //
    public double mse(double[][] X, double[] Y) {
        int N = X.length;
        double[] os = predict(X);
        double m = 0.0;
        for (int n = 0; n < N; n++) {
            m += Math.pow(os[n]-Y[n], 2) / N;
        }
        return m;
    }

    //
    public double r2(double[][] X, double[] Y) {
        int N = X.length;
        double[] os = predict(X);
        double meanOs = 0.0;
        double meanTs = 0.0;
        for (int n = 0; n < N; n++) {
            meanOs += os[n]/N;
            meanTs += Y[n]/N;
        }
        double s1 = 0.0;
        double s2 = 0.0;
        double s12 = 0.0;
        for (int n = 0; n < N; n++) {
            s1 += Math.pow(os[n]-meanOs, 2);
            s2 += Math.pow(Y[n]-meanTs, 2);
            s12 += (os[n]-meanOs)*(Y[n]-meanTs);
        }
        return Math.pow(s12, 2)/(s1*s2);
    }

}