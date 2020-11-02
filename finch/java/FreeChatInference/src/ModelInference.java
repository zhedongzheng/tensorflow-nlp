/*
└── FreeChatInference
	│
	├── data
	│   └── transformer_export/
	│   └── char.txt
	│   └── libtensorflow-1.14.0.jar
	│   └── tensorflow_jni.dll
	│
	└── src              
	    └── ModelInference.java
*/

import java.io.BufferedReader;
import java.io.File;
import java.io.FileInputStream;
import java.io.IOException;
import java.io.InputStreamReader;

import java.util.HashMap;
import java.util.Map;

import org.tensorflow.Graph;
import org.tensorflow.Session;
import org.tensorflow.Tensor;
import org.tensorflow.SavedModelBundle;

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
    
    public static int[][] processUtterance(String sentence) {
    	int sent_len = sentence.length();
    	int[][] indexArray = new int[1][sent_len];
    	String[] chars = sentence.split("");
    	
    	for (int i=0; i<sent_len; i++) {
    	    if (char2idx.containsKey(chars[i])) {
    	        indexArray[0][i] = char2idx.get(chars[i]);
    	    } else {
    	        indexArray[0][i] = char2idx.size();
    	    } // end if-else
    	} // end for
    	
    	return indexArray;
    } // end method processUtterance
	
    public static void main(String[] args) throws IOException {
        String query = "我想你了";
		
        readVocab("./data/char.txt");
		
        Session session = loadTFModel("C:\\Users\\zhedong.zheng\\eclipse-workspace\\FreeChatInference\\data\\transformer_export\\1587450204", "serve");
        
        int[][] indexArray = processUtterance(query);
        
        Tensor inputTensor = Tensor.create(indexArray);
        
        Tensor result = session.runner().feed("words:0", inputTensor).fetch("Decoder/strided_slice:0").run().get(0);
        
        int batchSize = (int) result.shape()[0];
        int timeStep = (int) result.shape()[1];
        int beamWidth = (int) result.shape()[2];
        
        int[][][] resultArray = new int[batchSize][timeStep][beamWidth];
        
        result.copyTo(resultArray);
        
        System.out.println("Input: " + query);
        for (int k=0; k<beamWidth; k++) {
            System.out.print("Rank_" + k + ": ");
            for (int j=0; j<timeStep; j++) {
                int c = resultArray[0][j][k];
                if (c != 0 && c != 2) {
                    if (idx2char.containsKey(c)) {
                        System.out.print(idx2char.get(resultArray[0][j][k]));
                    } else {
                        System.out.print("<unk>");
                    } // end if-else
                } // end if
            } // end for
            System.out.println();
        } // end for
    } // end main method
} // end class ModelInference
