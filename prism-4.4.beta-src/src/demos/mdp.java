//==============================================================================
//	
//	Copyright (c) 2017-
//	Authors:
//	* Dave Parker <d.a.parker@cs.bham.ac.uk> (University of Birmingham)
//	
//------------------------------------------------------------------------------
//	
//	This file is part of PRISM.
//	
//	PRISM is free software; you can redistribute it and/or modify
//	it under the terms of the GNU General Public License as published by
//	the Free Software Foundation; either version 2 of the License, or
//	(at your option) any later version.
//	
//	PRISM is distributed in the hope that it will be useful,
//	but WITHOUT ANY WARRANTY; without even the implied warranty of
//	MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
//	GNU General Public License for more details.
//	
//	You should have received a copy of the GNU General Public License
//	along with PRISM; if not, write to the Free Software Foundation,
//	Inc., 59 Temple Place, Suite 330, Boston, MA  02111-1307  USA
//	
//==============================================================================

package demos;

import java.io.FileOutputStream;
import java.io.PrintStream;

import java.io.File;
import java.io.FileNotFoundException;
import java.util.ArrayList;
import java.util.List;

import parser.ast.*;
import prism.PrismLangException;
import prism.PrismUtils;
import parser.type.*;

import prism.ModelType;
import prism.Prism;
import prism.PrismDevNullLog;
import prism.PrismException;
import prism.PrismLog;
import prism.Result;
import prism.UndefinedConstants;

import java.io.BufferedReader;
import java.io.FileReader;
import java.io.IOException;

import Jama.Matrix;


/**
 * An example class demonstrating how to control PRISM programmatically, through
 * the functions exposed by the class prism.Prism.
 * 
 * This shows how to load a model from a file and model check some properties,
 * either from a file or specified as a string, and possibly involving
 * constants.
 * 
 * See the README for how to link this to PRISM.
 */
public class mdp {
	public static int states = 0;
	public static int actions = 0;
	public static Matrix P_OPT;
	public static double [][][] MDP;
	public static Matrix starts;
	public static Matrix unsafe;
	public static int TRANSITIONS = 0;
	public static ArrayList<String> dtmc; 
	public static String dir;
	public static void main(String[] args) throws IOException, InterruptedException, PrismLangException {
		dir = args[0];
		new mdp().run(dir);
	}

	static final public void ConstantDef(ConstantList constantList, ArrayList<String> lines) {
		String sLastLine = lines.get(0), sCurrentLine = lines.get(1);
		for (String line : lines) {
			if (lines.indexOf(line) % 2 == 1) {
				sCurrentLine = line;
				try {
					if (sLastLine.equals("states"))
						states = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("actions"))
						actions = Integer.parseInt(sCurrentLine);
					constantList.addConstant(new ExpressionIdent(sLastLine),
							new ExpressionLiteral(TypeInt.getInstance(), Integer.parseInt(sCurrentLine)),
							TypeInt.getInstance());
				} catch (NumberFormatException e) {
					constantList.addConstant(new ExpressionIdent(sLastLine),
							new ExpressionLiteral(TypeDouble.getInstance(), Double.parseDouble(sCurrentLine)),
							TypeDouble.getInstance());
				}
			} else {
				sLastLine = line;
			}
		}
	}

	
	static final public void ParseMDP(ArrayList<String> lines) {
		MDP = new double[states][actions][states];
		for (int i = 0; i < lines.size(); i++) {
			String[] line = lines.get(i).split(" ");
			MDP[Integer.parseInt(line[0])]
					[Integer.parseInt(line[1])]
							[Integer.parseInt(line[2])]
									= Double.parseDouble(line[3]);
		}
	}

	
	
	static final public Module Module(String name, ConstantList constantList, FormulaList formulaList) {
		Module m = new Module(name);
		m.setName(name);
		m.addDeclaration(new Declaration("s", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				new ExpressionLiteral(TypeInt.getInstance(), states))));
		
		build_cmd(m);
		return m;
	}

	static final public void build_cmd(Module m) {
		dtmc = new ArrayList<String>();
		Command c = new Command();
		Updates us = new Updates();
		Update u = new Update();
		for (int i = 0; i < MDP.length; i++) {
			for(int j = 0; j < MDP[i].length; j++) {
				c = new Command();
				us = new Updates();
				u = new Update();
				c.setSynch("a" + Integer.toString(j));
				c.setSynchIndex(j);
				c.setGuard(new ExpressionLiteral(TypeBool.getInstance(),  "s=" + i));
				double p_total = 0.0;
				for (int k = 0; k < MDP[i][j].length; k++) {
					if(k == i) continue;
					double p = MDP[i][j][k];
					if(p > 0.0) {
						u.addElement(new ExpressionIdent("s"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(k)));
						us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), Double.toString(p)), u);
						u = new Update();
					}
					p_total += p;
				}
				if(p_total < 1.0) {
					u.addElement(new ExpressionIdent("s"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(i)));
					us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), Double.toString(1.0 - p_total)), u);
					u = new Update();
				}
				c.setUpdates(us);
				m.addCommand(c);
			}
			
		}
	} 
	

	
	static final public void run(String dir) throws InterruptedException, FileNotFoundException {
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			// PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine
			Prism prism = new Prism(mainLog);
			prism.initialise();

			ModulesFile mf = new ModulesFile();

			mf.setModelType(ModelType.MDP);

			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE =  dir + "data/state_space";
			String MDP = dir + "data/mdp";

			files.add(STATE_SPACE);
			files.add(MDP);
			ArrayList<String> lines = new ArrayList<String>();
			for (String file : files) {
				BufferedReader br = null;
				FileReader fr = null;
				try {
					fr = new FileReader(file);
					br = new BufferedReader(fr);
					String line;
					br = new BufferedReader(new FileReader(file));
					while ((line = br.readLine()) != null) {
						if(line.split(":")[line.split(":").length - 1].equals("0.0")) continue;
						lines.add(line);
					}
					//System.out.println(lines.size());
					if (file.equals(STATE_SPACE)) {
						ConstantDef(mf.getConstantList(), lines);
						lines.clear();
					}
					if (file.equals(MDP)) {
						ParseMDP(lines);
						lines.clear();
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			//Check_Transitions();
			
			Module m_opt = Module("grid_world", mf.getConstantList(), mf.getFormulaList());
			mf.addModule(m_opt);
			mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "s = " + Integer.toString(states - 2)));
			mf.tidyUp();
			System.out.println(mf);
			prism.loadPRISMModel(mf);
			PrintStream ps_console = System.out;
			PrintStream ps_file = new PrintStream(new FileOutputStream(
					new File(dir + "/mdp.pm")));
			System.setOut(ps_file);
			System.out.println(mf);
			
			
			System.setOut(ps_console);
			/**
			System.out.println(mf);
			PropertiesFile pf = prism.parsePropertiesFile(mf,
					new File(path + "/grid_world.pctl"));
			Result result = prism.modelCheck(pf, pf.getPropertyObject(0));
			System.out.println(result.getResult());
			**/

			System.exit(1);
		} catch (FileNotFoundException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		} catch (PrismException e) {
			System.out.println("Error: " + e.getMessage());
			System.exit(1);
		}

	}
}
