
/*

*/

public class Prepro {

    private int dim;
    private double[] vMin;
    private double[] vMax;

    public Prepro() {
        super();
    }

    // 
    public void initModel(int dimension, double[] dMin, double[] dMax) {
        dim = dimension;
        vMin = dMin;
        vMax = dMax;
    }

    // 
    public double[][] transform(double[][] data) {
        int N = data.length;
        double[][] newData = new double[N][dim];
        for (int i = 0; i < N; i++) {
            for (int j = 0; j < dim; j++) {
                newData[i][j] = (data[i][j]-vMin[j]) / (vMax[j]-vMin[j]);
            }
        }
        return newData;
    }

    // 
    public double[] oneTransform(double[] data) {
        double[] newData = new double[dim];
        for (int i = 0; i < dim; i++) {
            newData[i] = (data[i]-vMin[i]) / (vMax[i]-vMin[i]);
        }
        return newData;
    }

    // 
    public double[] reTransform(double[] data) {
        double[] newData = new double[dim];
        for (int i = 0; i < dim; i++) {
            newData[i] = data[i] * (vMax[i]-vMin[i]) + vMin[i];
        }
        return newData;
    }

}



