using BikeSharingDemand.BikeSharingDemandData;
using Microsoft.ML;
using Microsoft.ML.Data;
using System.Linq;

namespace BikeSharingDemand.ModelNamespace
{
    public sealed class Model
    {
        private readonly MLContext _mlContext;
        private IDataView _trainingDataView;
        private IEstimator<ITransformer> _algorythim;
        private ITransformer _trainedModel;
        private PredictionEngine<BikeSharingDemandSample, BikeSharingDemandPrediction> _predictionEngine;

        public string Name { get; private set; }

        public Model(MLContext mlContext, IEstimator<ITransformer> algorythm, string trainingDataLocation)
        {
            _mlContext = mlContext;
            _algorythim = algorythm;
            _trainingDataView = _mlContext.Data.LoadFromTextFile<BikeSharingDemandSample>(
                                        path: trainingDataLocation,
                                        hasHeader: true,
                                        separatorChar: ',');

            Name = algorythm.GetType().ToString().Split('.').Last();
        }

        public void BuildAndFit()
        {
            var pipeline = _mlContext.Transforms.CopyColumns(inputColumnName: "Count", outputColumnName: "Label")
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Season"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Year"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Holiday"))
                .Append(_mlContext.Transforms.Categorical.OneHotEncoding("Weather"))
                .Append(_mlContext.Transforms.Concatenate("Features",
                                                "Season",
                                                "Year",
                                                "Month",
                                                "Hour",
                                                "Weekday",
                                                "Weather",
                                                "Temperature",
                                                "Humidity",
                                                "Windspeed",
                                                "Casual"))
                .Append(_mlContext.Transforms.NormalizeMinMax("Features", "Features"))
                .AppendCacheCheckpoint(_mlContext)
                .Append(_algorythim);

            _trainedModel = pipeline.Fit(_trainingDataView);
            _predictionEngine = _mlContext.Model.CreatePredictionEngine<BikeSharingDemandSample, BikeSharingDemandPrediction>(_trainedModel);
        }
           
        public BikeSharingDemandPrediction Predict(BikeSharingDemandSample sample)
        {
            return _predictionEngine.Predict(sample);
        }

        public RegressionMetrics Evaluate(string testDataLocation)
        {
            var testDataView = _mlContext.Data.LoadFromTextFile<BikeSharingDemandSample>(
                                        path: testDataLocation,
                                        hasHeader: true,
                                        separatorChar: ',',
                                        allowQuoting: true,
                                        allowSparse: false);
            var predictions = _trainedModel.Transform(testDataView);
            return _mlContext.Regression.Evaluate(predictions, "Label", "Score");
        }

        public void SaveModel()
        {
            _mlContext.Model.Save(_trainedModel, _trainingDataView.Schema, "./BikeSharingDemandsModel.zip");
        }
    }
}
