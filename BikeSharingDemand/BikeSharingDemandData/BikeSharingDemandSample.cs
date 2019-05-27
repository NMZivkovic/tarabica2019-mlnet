using Microsoft.ML.Data;

namespace BikeSharingDemand.BikeSharingDemandData
{
    public class BikeSharingDemandSample
    {
        [LoadColumn(2)] public float Season;
        [LoadColumn(3)] public float Year;
        [LoadColumn(4)] public float Month;
        [LoadColumn(5)] public float Hour;
        [LoadColumn(6)] public bool Holiday;
        [LoadColumn(7)] public float Weekday;
        [LoadColumn(8)] public float Weather;
        [LoadColumn(10)] public float Temperature;
        [LoadColumn(12)] public float Humidity;
        [LoadColumn(13)] public float Windspeed;
        [LoadColumn(14)] public float Casual;
        [LoadColumn(16)] public float Count;
    }
}
