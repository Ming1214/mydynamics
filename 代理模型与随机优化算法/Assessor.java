
/*

*/

public class Assessor {

    public double predictOnePiece(double[] dInput) {
        return 0.0;
    }

    public double[] predict(double[][] dInputs) {
        int N = dInputs.length;
        double[] os = new double[N];
        for (int n = 0; n < N; n++) {
            os[n] = predictOnePiece(dInputs[n]);
        }
        return os;
    }
    
    private double mse(double[][] dInputs, double[] dTargets) {
        int N = dInputs.length;
        double[] os = predict(dInputs);
        double m = 0.0;
        for (int n = 0; n < N; n++) {
            m += Math.pow(os[n]-dTargets[n], 2) / N;
        }
        return m;
    }

    private double r2(double[][] dInputs, double[] dTargets) {
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

}



