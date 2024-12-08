
/*

*/

public class XGBCart extends Assessor {

    public class Node {
        public double value;
        public boolean leaf;
        public int splitDim;
        public double splitValue;
        public Node left;
        public Node right;
    }

    public class SplitResult {
        public int splitDim;
        public double splitValue;
        public double splitGain;
        public double[][] leftX;
        public double[] leftY;
        public double[][] rightX;
        public double[] rightY;
    }

    private int maxDepth;
    private int minSplitNum;
    private double minGain;
    private int intervalNum;
    private Node root;

    public int depth;

    private int dim;
    private double[] vMin;
    private double[] vMax;

    public XGBCart() {
        super();
    }

    public void initModel(int dDim, double[] dMin, double[] dMax) {
        dim = dDim;
        vMin = dMin;
        vMax = dMax;
    }

    public void train(double[][] trainX, double[] trainY, int max_depth, int min_split_num, double min_gain, int interval_num, double lamda) {
        maxDepth = max_depth;
        minSplitNum = min_split_num;
        minGain = min_gain;
        intervalNum = interval_num;
        depth = 0;
        root = grow(trainX, trainY, 0, lamda);
    }

    @Override
    public double predictOnePiece(double[] X) {
        return trace(X, root);
    }

    @Override
    public double[] predict(double[][] X) {
        int N = X.length;
        double[] os = new double[N];
        for (int n = 0; n < N; n++) {
            os[n] = predictOnePiece(X[n]);
        }
        return os;
    }
    
    private double trace(double[] X, Node n) {
        if (n.leaf) {
            return n.value;
        }
        if (X[n.splitDim] < n.splitValue) {
            return trace(X, n.left);
        }
        return trace(X, n.right);
    }

    private Node grow(double[][] X, double[] Y, int curDepth, double lamda) {
        Node n = new Node();
        n.value = sum(Y)/(lamda+Y.length);
        if ((Y.length <= minSplitNum) || (curDepth >= maxDepth)) {
            n.leaf = true;
            return n;
        }
        SplitResult splitR = split(X, Y, lamda);
        if (splitR.splitGain < minGain) {
            n.leaf = true;
            return n;
        }
        int nextDepth = curDepth + 1;
        if (nextDepth > depth) {
            depth = nextDepth;
        }
        n.leaf = false;
        n.splitDim = splitR.splitDim;
        n.splitValue = splitR.splitValue;
        n.left = grow(splitR.leftX, splitR.leftY, nextDepth, lamda);
        n.right = grow(splitR.rightX, splitR.rightY, nextDepth, lamda);
        return n;
    }

    private SplitResult split(double[][] X, double[] Y, double lamda) {
        int N = Y.length;
        SplitResult splitR = new SplitResult();
        splitR.splitGain = 0.0;
        for (int i = 0; i < dim; i++) {
            for (int j = 1; j < intervalNum; j++) {
                double splitv = j*((vMax[i]-vMin[i])/intervalNum)+vMin[i];
                int l = 0;
                for (int k = 0; k < N; k++) {
                    if (X[k][i] < splitv) {
                        l += 1;
                    }
                }
                if (l == 0) {
                    continue;
                }
                if (l == N) {
                    break;
                }
                double[] Y1 = new double[l];
                double[] Y2 = new double[N-l];
                int countL = 0;
                for (int k = 0; k < N; k++) {
                    if (X[k][i] < splitv) {
                        Y1[countL] = Y[k];
                        countL += 1;
                    } else {
                        Y2[k-countL] = Y[k];
                    }
                }
                double g = gain(Y1, Y2, lamda);
                if ((splitR.splitGain < g) && (g >= minGain)) {
                    splitR.splitGain = g;
                    splitR.splitDim = i;
                    splitR.splitValue = splitv;
                    splitR.leftY = Y1;
                    splitR.rightY = Y2;
                    splitR.leftX = new double[l][dim];
                    splitR.rightX = new double[N-l][dim];
                    countL = 0;
                    for (int k = 0; k < N; k++) {
                        if (X[k][i] < splitv) {
                            System.arraycopy(X[k], 0, splitR.leftX[countL], 0, dim);
                            countL += 1;
                        } else {
                            System.arraycopy(X[k], 0, splitR.rightX[k-countL], 0, dim);
                        }
                    }
                }
            }
        }
        return splitR;
    }

    private double gain(double[] Y1, double[] Y2, double lamda) {
        int N1 = Y1.length;
        int N2 = Y2.length;
        double y1 = sum(Y1);
        double y2 = sum(Y2);
        return Math.pow(y1, 2)/(lamda+N1)+Math.pow(y2, 2)/(lamda+N2)-Math.pow(y1+y2, 2)/(lamda+N1+N2);
    }

    private double sum(double[] Y) {
        int N = Y.length;
        double s = 0.0;
        for (int i = 0; i < N; i++) {
            s += Y[i];
        }
        return s;
    }

}



