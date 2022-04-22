using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class CSVData
    {
        public string[] Headers { get; set; }
        public string[]? LabelNames { get; set; }
        public double[][] Features { get; set; }
        public int[] Labels { get; set; }

        public CSVData(string[] headers, double[][] features, int[] labels, string[]? labelNames)
        {
            Headers = headers;
            Labels = labels;
            Features = features;
            LabelNames = labelNames;
        }
    }
}
