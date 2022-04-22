using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Adaboost
{
    public class CSVRead
    {
        public static CSVData ReadCSV(string dir)
        {
            List<string> headers = new List<string>();
            List<List<double>> features = new List<List<double>>();
            List<int> labels = new List<int>();
            List<string> labelNames = new List<string>();

            using (FileStream stream = File.OpenRead(dir))
            {
                using (StreamReader reader = new StreamReader(stream))
                {
                    int count = 0;

                    while (!reader.EndOfStream)
                    {
                        string? line = reader.ReadLine();
                        if (line == null) { continue; }
                        string[] splits = line.Split(',');
                        int numCols = splits.Length;

                        if (count == 0)
                        {
                            for (int i = 0; i < numCols - 1; i++)
                            {
                                features.Add(new List<double>());
                                headers.Add(splits[i]);
                            }
                        }
                        else
                        {
                            for (int i = 0; i < splits.Length - 1; i++)
                            {
                                _ = double.TryParse(splits[i], out double f);
                                features[i].Add(f);
                            }

                            labelNames.Add(splits[splits.Length - 1]);
                        }

                        count++;
                    }
                }
            }

            var distinctLabels = labelNames.Distinct().ToList();

            foreach (string label in labelNames)
            {
                for (int i = 0; i < distinctLabels.Count(); i++)
                {
                    if (label == distinctLabels[i])
                    {
                        labels.Add(i);
                        break;
                    }
                }
            }

            double[][] fArr = features.Select(x => x.ToArray()).ToArray();

            return new CSVData(headers.ToArray(), fArr, labels.ToArray(), labelNames.ToArray());
        }
    }
}
