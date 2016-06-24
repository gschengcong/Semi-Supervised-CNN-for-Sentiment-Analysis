import java.io.BufferedReader;
import java.io.BufferedWriter;
import java.io.File;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.FileWriter;
import java.io.IOException;

public class DataCleaning {

	public static void main(String[] args) throws IOException {
		// TODO Auto-generated method stub
		cleanTreeBankData("train.txt", "train_processed_data.txt");
		cleanTreeBankData("dev.txt", "dev_processed_data.txt");
		cleanTreeBankData("test.txt", "test_processed_data.txt");
		combineData("neg.txt", "pos.txt", "train_processed_data.txt");
	
	}
	
	// remove the parenthesis of in the treebank of stanford sentiment anlysis data
	public static void cleanTreeBankData(String inputFile, String outputFile) throws IOException{
		BufferedReader br = new BufferedReader(new FileReader(new File(inputFile)));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File(outputFile)));
		String line = br.readLine();
		int count = 0;
		while(line != null){
			String temp = (line.charAt(1) - '0')/3 + "";
			temp += line.toLowerCase().replaceAll("[0-9()]", " ").replaceAll("\\s+", " ") + "\n";
			bw.write(temp);
			line = br.readLine();
			count++;
		}
		br.close();
		bw.flush();
		bw.close();
		System.out.println(count);
	}
	
	// combine three training data
	public static void combineData(String file1, String file2, String file3) throws IOException{
		BufferedReader br1 = new BufferedReader(new FileReader(new File(file1)));
		BufferedReader br2 = new BufferedReader(new FileReader(new File(file2)));
		BufferedReader br3 = new BufferedReader(new FileReader(new File(file3)));
		BufferedWriter bw = new BufferedWriter(new FileWriter(new File("combined.txt")));
		String line1 = br1.readLine();
		String line2 = br2.readLine();
		String line3 = br3.readLine();
		while(line1 != null || line2 != null || line3 != null){
			if(line1 != null){
				bw.write("0 " + line1 + "\n");
				line1 = br1.readLine();
			}
			if(line2 != null){
				bw.write("1 " + line2 + "\n");
				line2 = br2.readLine();
			}
			if(line3 != null){
				bw.write(line3 + "\n");
				line3 = br3.readLine();
			}
		}
		br1.close();
		br2.close();
		br3.close();
		bw.flush();
		bw.close();
	}

}
