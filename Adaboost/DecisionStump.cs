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

        public double Train(double[] X, double[] W, int[] Y)
        {
            double[] thresholds = X.Distinct().ToArray();
            int[] predictions = new int[X.Length];
            double minError = double.MaxValue;
            double bestThreshold = Threshold;
            int bestPolarity = Parity;

            foreach(double t in thresholds)
            {
                double error = 0.0;

                for (int i = 0; i < X.Length; i++)
                {
                    predictions[i] = X[i] > t ? 1 : -1;
                    if(predictions[i] != Y[i]) { error += W[i]; }
                }

                if(error < minError)
                {
                    minError = error;
                    bestThreshold = t;
                    bestPolarity = 1;
                }

                error = 0;

                for (int i = 0; i < X.Length; i++)
                {
                    predictions[i] = X[i] > t ? -1 : 1;
                    if (predictions[i] != Y[i]) { error += W[i]; }
                }

                if (error < minError)
                {
                    minError = error;
                    bestThreshold = t;
                    bestPolarity = -1;
                }
            }

            this.Threshold = bestThreshold;
            this.Parity = bestPolarity;
            this.Alpha = 0.5 * Math.Log((1 - minError) / minError);

            return minError;
        }
    }
}
