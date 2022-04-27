using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    internal class DecisionStump
    {
        public int Parity { get; private set; }
        public int FeatureIndex { get; private set; }
        public double Threshold { get; private set; }
        public double Alpha { get; private set; }

        public DecisionStump(int idx)
        {
            Parity = 1;
            FeatureIndex = idx;
            Threshold = 0;
            Alpha = 0;
        }

        public int[] Predict(double[] X)
        {
            int[] Y = new int[X.Length];

            for (int i = 0; i < X.Length; i++)
            {
                Y[i] = X[i] > Threshold ? 1 * Parity : -1 * Parity;
            }

            return Y;
        }

        public int Predict(double X)
        {
            int Y = X > Threshold ? 1 * Parity : -1 * Parity;
            return Y;
        }

        public int[] Predict(AdaData[] data)
        {
            int[] Y = new int[data.Length];

            for (int i = 0; i < data.Length; i++)
            {
                Y[i] = data[i][FeatureIndex] > Threshold ? 1 * Parity : -1 * Parity;
            }

            return Y;
        }

        public int Predict(AdaData data)
        {
            return data[FeatureIndex] > Threshold ? 1 * Parity : -1 * Parity;
        }

        public double Train(AdaData[] data, double[] W)
        {
            double[] thresholds = data.Select(x => x[FeatureIndex]).Distinct().ToArray();
            double minError = double.MaxValue;
            double bestThreshold = Threshold;
            int bestPolarity = Parity;

            foreach (double t in thresholds)
            {
                double error = 0.0;
                int p = 1;

                for (int i = 0; i < data.Length; i++)
                {
                    int yP = data[i][FeatureIndex] > t ? 1 : -1;
                    if (yP != data[i].Label) { error += W[i]; }
                }

                if(error > 0.5)
                {
                    p = -1;
                    error = 1.0 - error;
                }

                if (error < minError)
                {
                    minError = error;
                    bestThreshold = t;
                    bestPolarity = p;
                }
            }

            this.Threshold = bestThreshold;
            this.Parity = bestPolarity;
            this.Alpha = 0.5 * Math.Log((1 - minError) / minError);

            return minError;
        }
    }
}
