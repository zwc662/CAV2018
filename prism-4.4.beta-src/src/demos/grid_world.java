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
public class grid_world {
	public static int states = 0;
	public static int actions = 0;
	public static Matrix P_OPT;
	public static double[][] policy;

	public static double[][] trans;
	public static int TRANSITIONS = 0;
	public static ArrayList<String> dtmc; 
	public static void main(String[] args) throws IOException, InterruptedException, PrismLangException {
		//System.out.println(args[0]);
		new grid_world().run(args[0]);
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

	static final public void ParsePolicy(Module m, ArrayList<String> lines) {
		policy = new double[states][states];
		trans = new double[states][states];
		int from = 0;
		for (int i = 0; i < lines.size(); i++) {
			String[] line = lines.get(i).split(" ");
			policy[Integer.parseInt(line[0])][Integer.parseInt(line[1])] = Double.parseDouble(line[2]);

		}
		trans = check_policy(policy);
		build_cmd(m);
	}
	
	static final public double[][] check_policy(double [][] policy) {
		trans = new double[states][states];
		for (int i = 0; i < states; i++) {
			for (int j = 0; j < states; j++) {
				double p = policy[i][j];
				trans[i][j] = p;
			}
		}
		/*	
		int itr = 0;
		boolean [] trim = new boolean[states];
		boolean done = false;
		while(!done) {
			done = true;
			for (int i = 0; i < states; i++) {
				double p_tot = 0.0;
				for (int j = 0; j < states; j++) {
					if(trim[i]) trans[i][j] = policy[i][j] * Math.pow(0.9, itr);
					if(policy[i][j] > 0.0)	{
						trans[i][j]= (double)((int)(1 * trans[i][j])) / 1;
					}
					p_tot += trans[i][j];
				}
				if(p_tot > 1.0) trim[i] = true;
				else {
					trim[i] = false;
					trans[i][i] += 1.0 - p_tot;
				}
				done = done && trim[i];
			}
			itr++;
			System.out.println(trans[0][897]);
		}
		*/
		return trans;
	}
	

	static final public Module Module(String name, ConstantList constantList, FormulaList formulaList) {
		Module m = new Module(name);
		m.setName(name);
		m.addDeclaration(new Declaration("s", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				new ExpressionLiteral(TypeInt.getInstance(), states - 1))));
		return m;
	}

	static final public void build_cmd(Module m) {
		dtmc = new ArrayList<String>();
		for(int i = 0; i < states; i++) {
			Command c = new Command();
			Updates us = new Updates();
			Update u = new Update();
			c.setGuard(new ExpressionLiteral(TypeBool.getInstance(),  "(s= "+ i + ") = true"));
			double p_tot = 0.0;
			for (int j = 0; j < states; j++) {
				double p = policy[i][j];
				if(p > 0.0) {
					u.addElement(new ExpressionIdent("s"), new ExpressionLiteral(TypeInt.getInstance(), Integer.toString(j)));
					us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), Double.toString(p)), u);
					TRANSITIONS = TRANSITIONS + 1;
					//dtmc.add(Integer.toString(i) + ' ' + Integer.toString(j) + ' ' + Double.toString(p));
					dtmc.add(String.format("%d %d %f", i,j, p));
					u = new Update();
				}
			}
			c.setUpdates(us);
			m.addCommand(c);
		}
	}
	
	static final public void Write_DTMC() {
		dtmc.add(0, "STATES " + Integer.toString(states));
		dtmc.add(1, "TRANSITIONS " + Integer.toString(TRANSITIONS));
		dtmc.add(2, "INITIAL " + Integer.toString(policy.length- 2));
		dtmc.add(3, "TARGET " + Integer.toString(policy.length - 1));
	}
	
	static final public void run(String path) throws InterruptedException, FileNotFoundException {
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			// PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine
			Prism prism = new Prism(mainLog);
			prism.initialise();

			ModulesFile mf = new ModulesFile();

			mf.setModelType(ModelType.DTMC);
			Module m_opt;
			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE = path + "/data/state_space";
			String OPT_POLICY = path + "/data/optimal_policy";
			
			files.add(STATE_SPACE);
			files.add(OPT_POLICY);
			
			ArrayList<String> lines = new ArrayList<String>();
			for (String file : files) {
				BufferedReader br = null;
				FileReader fr = null;
				try {
					fr = new FileReader(file);
					br = new BufferedReader(fr);
					String line;
					br = new BufferedReader(new FileReader(file));
					
					if (file.equals(STATE_SPACE)) {
						while ((line = br.readLine()) != null) {
							lines.add(line);
						}
						ConstantDef(mf.getConstantList(), lines);
						lines.clear();
					}

					if (file.equals(OPT_POLICY)) {
						m_opt = Module("grid_world", mf.getConstantList(), mf.getFormulaList());
						int episode = states;
						while ((line = br.readLine()) != null) {
							lines.add(line);
							}
						ParsePolicy(m_opt, lines);
						mf.addModule(m_opt);		
					 }
					
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			//Check_Transitions();
			
			
			
			Write_DTMC();
			//mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "s = 0 & y = 0"));
			mf.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(),
					"s = " + Integer.toString(states-2)));
			mf.tidyUp();
			//System.out.println(mf);
			prism.loadPRISMModel(mf);
			PropertiesFile pf = prism.parsePropertiesString(mf, "P=? [F<" 
																+ Integer.toString(states) 
																+ " s=" 
																+ Integer.toString(states - 1) 
																+ "]");

			PrintStream ps_console = System.out;
			PrintStream ps_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.pm")));
			System.setOut(ps_file);
			System.out.println(mf);
			
			PrintStream pctl_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.pctl")));
			System.setOut(pctl_file);
			System.out.println(pf);
			
			PrintStream dtmc_file = new PrintStream(new FileOutputStream(
					new File(path + "/grid_world.dtmc")));
			System.setOut(dtmc_file);
			for(String i:dtmc) {
				System.out.println(i);
			}
			
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