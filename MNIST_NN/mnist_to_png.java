import java.io.File;
import java.util.Scanner;
import java.awt.image.BufferedImage;
import java.awt.image.WritableRaster;
import javax.imageio.ImageIO;

class mnist_to_png{
	public static void main(String[] args) throws Exception{
		final int IMG_WIDTH = 14, IMG_HEIGHT = 14; 
		final String DELIMITER = ",";
		int count[] = new int[10];
		Scanner sc = new Scanner(new File(args[0]));
		for (int i = 0; i < 10; ++i){
			count[i] = 0;
		}
		while (sc.hasNextLine()){
			Scanner l = new Scanner(sc.nextLine()).useDelimiter(DELIMITER);
			int label;
			int[] data = new int[IMG_HEIGHT * IMG_WIDTH]; 
			BufferedImage img = new BufferedImage(IMG_WIDTH, IMG_HEIGHT, BufferedImage.TYPE_BYTE_GRAY);
			for (int i = 0; i < IMG_HEIGHT * IMG_WIDTH; ++i){
				data[i] = 255 - l.nextInt() << 5;   //converting from 0-7 scale to 0-255 scale
			}
			label = l.nextInt();
			img.getRaster().setPixels(0, 0, IMG_WIDTH, IMG_HEIGHT, data);
			ImageIO.write(img, "png", new File(label + "_" + count[label]++ + ".png"));
		}
	}
}
