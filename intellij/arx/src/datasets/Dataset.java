package datasets;

import org.deidentifier.arx.ARXConfiguration;
import org.deidentifier.arx.Data;

import java.io.IOException;
import java.util.Map;

public abstract class Dataset{
    protected String outPath;
    protected String inPath;

    protected String statisticsOutPath;

    public abstract ARXConfiguration getConfig(Map<String,Double> configParams);

    public abstract Data getData(boolean generateHierarchies) throws IOException;

    public Data getData() throws IOException{
        return this.getData(true);
    }

    public abstract Map<String,double[]> getParams();

    public String getOutPath() {
        return outPath;
    }

    public String getStatisticsOutPath() {
        return statisticsOutPath;
    }
}
