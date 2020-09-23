package YF;
import java.io.BufferedReader;
import java.io.IOException;
import java.io.InputStreamReader;


public class API {
	public void LSTMTraining(String str) {
        Process proc;
        String command = "python3 /home/SaveAndLoad/save.py "+str;
        try {
            proc = Runtime.getRuntime().exec(command);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } 
	}
	public void LSTMPrediction(String str) {
        Process proc;
        String command = "python3 /home/SaveAndLoad/load.py "+str;

        try {
            proc = Runtime.getRuntime().exec(command);
            BufferedReader in = new BufferedReader(new InputStreamReader(proc.getInputStream()));
            String line = null;
            while ((line = in.readLine()) != null) {
                System.out.println(line);
            }
            in.close();
            proc.waitFor();
        } catch (IOException e) {
            e.printStackTrace();
        } catch (InterruptedException e) {
            e.printStackTrace();
        } 
	}
	public void test() {
		System.out.println("hshkdjhd");
	}
}
