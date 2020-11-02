import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;
import java.util.Arrays;
import java.util.HashMap;
import java.util.Map;

import org.tensorflow.SavedModelBundle;
import org.tensorflow.Session;
import org.tensorflow.Tensor;

public class ModelInference {
    private static Map<String, Integer> char2idx = new HashMap<>();
    private static Map<Integer, String> idx2char = new HashMap<>();

    static {
        try {
            System.load("C:\\Users\\zhedong.zheng\\eclipse-workspace\\FreeChatInference\\data\\tensorflow_jni.dll");
        } catch (UnsatisfiedLinkError e) {
            System.err.println("Native code library failed to load.\n" + e);
            System.exit(1);
        }
    }	

    private static void readVocab(String pathname) throws IOException{
        BufferedReader br = new BufferedReader(new InputStreamReader(new FileInputStream(new File(pathname))));
        
        int count = 0;
        String line;
        
        while ((line = br.readLine()) != null) {
            char2idx.put(line, count);
            idx2char.put(count, line);
            count += 1;
        }
        br.close();
        System.out.println("Vocab Length: " + char2idx.size());
    } // end method readVocab    
    
    private static Session loadTFModel(String pathname, String tag) throws IOException{
        SavedModelBundle modelBundle = SavedModelBundle.load(pathname, tag);
        Session session = modelBundle.session();
        return session;
    } // end method loadTFModel

    public static int[][] processQuery(String input) {
        String[] utts = input.split("<SEP>");
        String query = utts[utts.length-1];
    	
        int sent_len = query.length();
        int[][] queryArray = new int[1][sent_len];
        String[] chars = query.split("");
    	
        for (int i=0; i<sent_len; i++) {
            if (char2idx.containsKey(chars[i])) {
                queryArray[0][i] = char2idx.get(chars[i]);
            } else {
                queryArray[0][i] = char2idx.size();
            } // end if-else
        } // end for
    	
    	return queryArray;
    } // end method processQuery
    
    public static int[][][] processHistory(String input) {
        String[] utts = input.split("<SEP>");
        String[] history = Arrays.copyOfRange(utts, 0, utts.length-1);
    	
        int maxlen = 0;
        for (String hist: history) {
            if (hist.length() > maxlen)
                maxlen = hist.length();
        }
    	
        int[][][] historyArray = new int[1][history.length][maxlen];
    	
        for (int i=0; i<history.length; i++) {
            String[] chars = history[i].split("");
            for (int j=0; j<history[i].length(); j++) {
                if (char2idx.containsKey(chars[i])) {
                    historyArray[0][i][j] = char2idx.get(chars[j]);
                } else {
                    historyArray[0][i][j] = char2idx.size();
                } // end if-else
            }
        }
    	
        return historyArray;
    } // end method processHistory
    
    public static void main(String[] args) throws IOException {
    	String input = "上海的天气怎么样<SEP>还行吧<SEP>北京的呢";
    	
    	readVocab("./data/char.txt");
    	
    	Session session = loadTFModel("C:\\Users\\zhedong.zheng\\eclipse-workspace\\MultiDialogInference\\data\\baseline_lstm_greedy_export\\1587959473", "serve");
    	
    	int[][] queryArray = processQuery(input);
    	int[][][] historyArray = processHistory(input);
    	
    	Tensor queryTensor = Tensor.create(queryArray);
    	Tensor historyTensor = Tensor.create(historyArray);
    	
    	Tensor result = session.runner().feed("query:0", queryTensor).feed("history:0", historyTensor).fetch("Decoder/decoder/transpose_1:0").run().get(0);
        int batchSize = (int) result.shape()[0];
        int timeStep = (int) result.shape()[1];
        
        int[][] resultArray = new int[batchSize][timeStep];
        
        result.copyTo(resultArray);
        
        System.out.println("Input: " + input);
        System.out.print("Output: ");
    	for (int j=0; j<timeStep; j++) {
        	int c = resultArray[0][j];
        	if (c != 0 && c != 2) {
        		if (idx2char.containsKey(c)) {
        			System.out.print(idx2char.get(resultArray[0][j]));
        		} else {
        			System.out.print("<unk>");
        		} // end if-else
        	} // end if
    	} // end for
    } // end main method
} // end class ModelInference
