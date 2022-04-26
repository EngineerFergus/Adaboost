using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class AdaData : LabeledData
    {
        public AdaData(double[] features, int Label) : base(features, Label)
        {
            if(Label != -1 && Label != 1)
            {
                throw new ArgumentException($"Cannot create instance of {nameof(AdaData)} with label " +
                    $"value that is not -1 or 1.");
            }
        }
    }
}
