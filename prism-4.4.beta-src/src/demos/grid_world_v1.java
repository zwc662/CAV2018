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
public class grid_world_v1 {
	public static int x_max = 0;
	public static int y_max = 0;
	public static int x_l_0 = 0;
	public static int y_l_0 = 0;
	public static int x_h_0 = 0;
	public static int y_h_0 = 0;
	public static int x_h_1 = 0;
	public static int y_h_1 = 0;
	public static double p = 0.0;
	public static Matrix P_bad;
	public static Matrix P_good;
	public static Matrix P_OPT;
	public static Matrix P_DEMO;
	public static Matrix policy;

	public static void main(String[] args) throws IOException, InterruptedException, PrismLangException {
		// Process proc = Runtime.getRuntime().exec("python
		// /Users/weichaozhou/Documents/Safe_AI_MDP/workspace/grid_world/cirl/run.py");
		// proc.waitFor();
		new grid_world_v1().run();
	}

	static final public void ConstantDef(ConstantList constantList, ArrayList<String> lines) {
		String sLastLine = lines.get(0), sCurrentLine = lines.get(1);
		for (String line : lines) {
			if (lines.indexOf(line) % 2 == 1) {
				sCurrentLine = line;
				try {
					if (sLastLine.equals("x_max"))
						x_max = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("y_max"))
						y_max = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("x_l_0"))
						x_l_0 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("y_l_0"))
						y_l_0 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("x_h_0"))
						x_h_0 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("y_h_0"))
						y_h_0 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("x_h_1"))
						x_h_1 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("y_h_1"))
						y_h_1 = Integer.parseInt(sCurrentLine);
					else if (sLastLine.equals("p"))
						p = Double.parseDouble(sCurrentLine);

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
		constantList.addConstant(new ExpressionIdent("e"), new ExpressionLiteral(TypeDouble.getInstance(), 2.72),
				TypeDouble.getInstance());
		constantList.addConstant(new ExpressionIdent("x_min"), new ExpressionLiteral(TypeInt.getInstance(), 0),
				TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_min"), new ExpressionLiteral(TypeInt.getInstance(), 0),
				TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("x_init"), new ExpressionLiteral(TypeInt.getInstance(), 0),
				TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_init"), new ExpressionLiteral(TypeInt.getInstance(), 0),
				TypeInt.getInstance());

		// System.out.println(constantList);
	}

	static final public void Prop_ConstantDef(ConstantList constantList, int x_init, int y_init, int x_end, int y_end) {
		// constantList.addConstant(new ExpressionIdent("x_init"), new
		// ExpressionLiteral(TypeInt.getInstance(), x_init),
		// TypeInt.getInstance());
		// constantList.addConstant(new ExpressionIdent("y_init"), new
		// ExpressionLiteral(TypeInt.getInstance(), y_init),
		// TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("x_end"), new ExpressionLiteral(TypeInt.getInstance(), x_end),
				TypeInt.getInstance());
		constantList.addConstant(new ExpressionIdent("y_end"), new ExpressionLiteral(TypeInt.getInstance(), y_end),
				TypeInt.getInstance());

	}

	static final public void Prop_PropertyDef(PropertiesFile pf_opt) {
		String property1 = "Pmin=?[F (x=x_end & y=y_end)]";
		pf_opt.addProperty(new Property(new ExpressionLiteral(TypeDouble.getInstance(), property1)));
	}

	static final public void ParsePolicy(ArrayList<String> lines) {
		policy = new Matrix(new double[lines.size()][lines.get(0).split(":").length]);
		for (int i = 0; i < lines.size(); i++) {
			String[] actions = lines.get(i).split(":");
			for (int j = 0; j < actions.length; j++) {
				policy.set(i, j, Double.parseDouble(actions[j]));
			}
		}
	}

	static final public void FormulaDef(FormulaList formulaList, Matrix policy) {
		 String stay = new String("("), right = new String("("), down = new String("("), left = new String("("), up = new String("("), sink = new String("(");
		 for(int y = 0; y < policy.getColumnDimension(); y++) { 
			 for(int x = 0; x < policy.getRowDimension(); x++)	{
				 switch((int)policy.get(y, x))	{
			      case 0:	stay = build_expr(stay, x, y);	break;
			      case 1:	right = build_expr(right, x, y);	break;
			      case 2:	down = build_expr(down, x, y);	break;
			      case 3:	left = build_expr(left, x, y);	break;
			      case 4:	up = build_expr(up, x, y);	break;
			      default:	sink = build_expr(sink, x, y);
			        break;
			     }
			 }
		 }
		 
		 if(stay.equals("(")) stay = "false";
		 else	stay = stay + ")";
		 if(right.equals("(")) right = "false";
		 else right = right + ")";
		 if(down.equals("(")) down = "false";
		 else down = down + ")";
		 if(left.equals("(")) left= "false";
		 else left = left + ")";
		 if(up.equals("(")) up = "false";
		 else up = up + ")";
		 if(sink.equals("(")) sink = "false";
		 else sink = sink + ")";
		 
		 Expression stay_expr = new ExpressionLiteral(TypeBool.getInstance(), stay);
		 Expression right_expr = new ExpressionLiteral(TypeBool.getInstance(), right);
		 Expression down_expr = new ExpressionLiteral(TypeBool.getInstance(), down);
		 Expression left_expr = new ExpressionLiteral(TypeBool.getInstance(), left);
		 Expression up_expr = new ExpressionLiteral(TypeBool.getInstance(), up);
		 //Expression sink_expr = new ExpressionLiteral(TypeBool.getInstance(), sink);
		 if(formulaList.size() == 0) {
		 	formulaList.addFormula(new ExpressionIdent("stay"), stay_expr);
		 	formulaList.addFormula(new ExpressionIdent("right"), right_expr);
		 	formulaList.addFormula(new ExpressionIdent("down"), down_expr);
		 	formulaList.addFormula(new ExpressionIdent("left"), left_expr);
		 	formulaList.addFormula(new ExpressionIdent("up"), up_expr);
		 } else {
			 formulaList.setFormula(formulaList.getFormulaIndex("stay"), stay_expr);
			 formulaList.setFormula(formulaList.getFormulaIndex("right"), right_expr);
			 formulaList.setFormula(formulaList.getFormulaIndex("down"), down_expr);
			 formulaList.setFormula(formulaList.getFormulaIndex("left"), left_expr);
			 formulaList.setFormula(formulaList.getFormulaIndex("up"), up_expr);
		 }
		 //formulaList.addFormula(new ExpressionIdent("sink"), sink_expr);
		 //System.out.println(formulaList);
	 }

	static final public String build_expr(String action, int x, int y) {
		if (action.equals("(")) {
			action = action + "(x=" + String.valueOf(x) + " & y=" + String.valueOf(y) + ")";
		} else {
			action = action + " | (x=" + String.valueOf(x) + " & y=" + String.valueOf(y) + ")";
		}
		return action;
	}

	static final public Module Module(String name, ConstantList constantList, FormulaList formulaList) {
		Module m = new Module(name);
		m.setName(name);
		m.addDeclaration(new Declaration("x", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				constantList.getConstant(constantList.getConstantIndex("x_max")))));
		m.addDeclaration(new Declaration("y", new DeclarationInt(new ExpressionLiteral(TypeInt.getInstance(), 0),
				constantList.getConstant(constantList.getConstantIndex("y_max")))));
		build_cmd(m, constantList, formulaList);
		return m;
	}

	static final public void build_cmd(Module m, ConstantList constantList, FormulaList formulaList) {
		for (int i = 0; i < formulaList.size(); i++) {
			Command c = new Command();
			Updates us = new Updates();
			Update u = new Update();
			c.setSynch(formulaList.getFormulaName(i));
			c.setSynchIndex(i);
			c.setGuard(new ExpressionLiteral(TypeBool.getInstance(), formulaList.getFormulaNameIdent(i) + "=true"));

			if (i == 0) {
				u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
				u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));
				;
				us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "1"), u);
				c.setUpdates(us);
				m.addCommand(c);
				continue;
			}
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));
			;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "(x+1>x_max?x-1:x+1)"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));
			;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "(y+1>y_max?y-1:y+1)"));
			;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "(x-1<x_min?x+1:x-1)"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "y"));
			;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);
			u = new Update();
			u.addElement(new ExpressionIdent("x"), new ExpressionLiteral(TypeInt.getInstance(), "x"));
			u.addElement(new ExpressionIdent("y"), new ExpressionLiteral(TypeInt.getInstance(), "(y-1<y_min?y+1:y-1)"));
			;
			us.addUpdate(new ExpressionLiteral(TypeDouble.getInstance(), "(1-p)/4"), u);

			us.setProbability(i, new ExpressionLiteral(TypeDouble.getInstance(), "p"));
			u = new Update();
			c.setUpdates(us);
			m.addCommand(c);
		} /**
			 * Command c = new Command(); Updates us = new Updates(); Update u =
			 * new Update(); c.setSynch(formulaList.getFormulaName(0));
			 * c.setSynchIndex(0); c.setGuard(new
			 * ExpressionLiteral(TypeBool.getInstance(),
			 * formulaList.getFormulaNameIdent(0)+"=true")); u.addElement(new
			 * ExpressionIdent("x"), new
			 * ExpressionLiteral(TypeInt.getInstance(), "x")); u.addElement(new
			 * ExpressionIdent("y"), new
			 * ExpressionLiteral(TypeInt.getInstance(), "y"));; us.addUpdate(new
			 * ExpressionLiteral(TypeDouble.getInstance(), "1"), u);
			 * m.addCommand(c);
			 **/
	}

	static final public void set_good(Prism prism, ModulesFile mf_good, double epsilon)
			throws FileNotFoundException, PrismException {
		P_good = new Matrix(new double[y_max + 1][x_max + 1]);
		policy = new Matrix(new double[y_max + 1][x_max + 1]);
		for (int i = 0; i <= x_max; i++) {
			for (int j = 0; j <= y_max; j++) {
				policy.set(i, j, 1.0);
			}
		}
		policy.set(y_l_0, x_l_0, 0.0);
		policy.set(y_h_0, x_h_0, 0.0);
		policy.set(y_h_1, x_h_1, 0.0);

		FormulaDef(mf_good.getFormulaList(), policy);
		Module m_good = Module("grid_good", mf_good.getConstantList(), mf_good.getFormulaList());
		mf_good.addModule(m_good);

		Double diff = Double.MAX_VALUE;
		while (diff > epsilon) {
			diff = 0.0;
			
			for (int i = 0; i <= y_max; i++) {
				for (int j = 0; j <= x_max; j++) {
					mf_good.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "y = " + Integer.toString(i) + "& x = " + Integer.toString(j)));
					mf_good.tidyUp();
					PrintStream ps_console = System.out;
					PrintStream ps_file = new PrintStream(new FileOutputStream(new File("//home/zekunzhou/workspace/Safety-AI-MDP/prism-4.3.1-src/src/demos/grid_good.pm")));
					System.setOut(ps_file);
					//System.out.println(mf_good);
					System.setOut(ps_console);
					//System.out.println(mf_good);
					ModulesFile mf = prism.parseModelFile(new File("//home/zekunzhou/workspace/Safety-AI-MDP/prism-4.3.1-src/src/demos/grid_good.pm"));
					prism.loadPRISMModel(mf);
					PropertiesFile propertiesFile = prism.parsePropertiesString(mf, "P=? [F x = " + Integer.toString(x_l_0) + " & y = " + Integer.toString(y_l_0) + "]");
					Result result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
					if (Math.abs(P_good.get(i, j) - (double) result.getResult()) > diff)
						diff = Math.abs(P_good.get(i, j) - (double) result.getResult());
					P_good.set(i, j, (double) result.getResult());
				}
			}
			
			for (int i = 0; i <= y_max; i++) {
				for (int j = 0; j <= x_max; j++) {
					double good = P_good.get(i, j);
					double chaos = (1-p) * 0.25 * (P_good.get(i, x_max - Math.abs(x_max - j - 1)) + P_good.get(y_max - Math.abs(y_max - i - 1), j) + P_good.get(i, Math.abs(j - 1)) + P_good.get(Math.abs(i - 1), j));
					if (chaos + p * P_good.get(i, x_max - Math.abs(x_max - j - 1)) < good) {
						policy.set(i, j, 1);
						good = chaos + p * P_good.get(i, x_max - Math.abs(x_max - j - 1));
					} else if (chaos + p * P_good.get(y_max - Math.abs(y_max - i - 1), j) < good) {
						policy.set(i, j, 2);
						good = chaos + p * P_good.get(y_max - Math.abs(y_max - i - 1), j);
					} else if (chaos + p * P_good.get(i, Math.abs(j - 1)) < good) {
						policy.set(i, j, 3);
						good = chaos + p * P_good.get(i, Math.abs(j - 1));
					} else if (chaos + p * P_good.get(Math.abs(i - 1), j) < good) {
						policy.set(i, j, 4);
						good = chaos + p * P_good.get(Math.abs(i - 1), j);
					} 
					policy.set(y_l_0, x_l_0, 0);
					policy.set(y_h_0, x_h_0, 0);
					policy.set(y_h_1, x_h_1, 0);
					
					
					
				}
			}
			
			FormulaDef(mf_good.getFormulaList(), policy);
			
		}
	}

	static final public void set_bad(Prism prism, ModulesFile mf_bad, double epsilon)
			throws FileNotFoundException, PrismException {
		P_bad = new Matrix(new double[y_max + 1][x_max + 1]);
		policy = new Matrix(new double[y_max + 1][x_max + 1]);
		for (int i = 0; i <= x_max; i++) {
			for (int j = 0; j <= y_max; j++) {
				policy.set(i, j, 1.0);
			}
		}
		policy.set(y_l_0, x_l_0, 0.0);
		policy.set(y_h_0, x_h_0, 0.0);
		policy.set(y_h_1, x_h_1, 0.0);
		
		FormulaDef(mf_bad.getFormulaList(), policy);
		Module m_bad = Module("grid_bad", mf_bad.getConstantList(), mf_bad.getFormulaList());
		mf_bad.addModule(m_bad);

		Double diff = Double.MAX_VALUE;
		while (diff > epsilon) {
			diff = 0.0;

			for (int i = 0; i <= y_max; i++) {
				for (int j = 0; j <= x_max; j++) {
					mf_bad.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(),
							"y = " + Integer.toString(i) + "& x = " + Integer.toString(j)));
					mf_bad.tidyUp();
					PrintStream ps_console = System.out;
					PrintStream ps_file = new PrintStream(new FileOutputStream(new File("//home/zekunzhou/workspace/Safety-AI-MDP/prism-4.3.1-src/src/demos/grid_bad.pm")));
					System.setOut(ps_file);
					//System.out.println(mf_bad);
					System.setOut(ps_console);
					ModulesFile mf = prism.parseModelFile(new File("//home/zekunzhou/workspace/Safety-AI-MDP/prism-4.3.1-src/src/demos/grid_bad.pm"));
					prism.loadPRISMModel(mf);
					
					PropertiesFile propertiesFile = prism.parsePropertiesString(mf, "P=? [true U<64 x = " + Integer.toString(x_l_0) + " & y = " + Integer.toString(y_l_0) + "]");
					Result result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
					if (Math.abs(P_bad.get(i, j) - (double) result.getResult()) > diff)
						diff = Math.abs(P_bad.get(i, j) - (double)result.getResult());
						P_bad.set(i, j,  (double) result.getResult());
				}
			}
			
			for (int i = 0; i <= y_max; i++) {
				for (int j = 0; j <= x_max; j++) {
					double bad = P_bad.get(i, j);
					double chaos = (1-p) * 0.25 * (P_bad.get(i, x_max - Math.abs(x_max - j - 1)) + P_bad.get(y_max - Math.abs(y_max - i - 1), j) + P_bad.get(i, Math.abs(j - 1)) + P_bad.get(Math.abs(i - 1), j));
					if (chaos + p * P_bad.get(i, x_max - Math.abs(x_max - j - 1)) > bad) {
						policy.set(i, j, 1);
						bad = chaos + p * P_bad.get(i, x_max - Math.abs(x_max - j - 1));
					} else if (chaos + p * P_bad.get(y_max - Math.abs(y_max - i - 1), j) > bad) {
						policy.set(i, j, 2);
						bad = chaos + p * P_bad.get(y_max - Math.abs(y_max - i - 1), j);
					} else if (chaos + p * P_bad.get(i, Math.abs(j - 1)) > bad) {
						policy.set(i, j, 3);
						bad = chaos + p * P_bad.get(i, Math.abs(j - 1));
					} else if (chaos + p * P_bad.get(Math.abs(i - 1), j) > bad) {
						policy.set(i, j, 4);
						bad = chaos + p * P_bad.get(Math.abs(i - 1), j);
					}
					
					policy.set(y_l_0, x_l_0, 0);
					policy.set(y_h_0, x_h_0, 0);
					policy.set(y_h_1, x_h_1, 0);
					
				}
			}
			
			
			FormulaDef(mf_bad.getFormulaList(), policy);
		}
	}

	static final public void run() throws InterruptedException, FileNotFoundException {
		try {
			// Create a log for PRISM output (hidden or stdout)
			PrismLog mainLog = new PrismDevNullLog();
			// PrismLog mainLog = new PrismFileLog("stdout");

			// Initialise PRISM engine
			Prism prism = new Prism(mainLog);
			prism.initialise();

			ModulesFile mf_opt = new ModulesFile();
			ModulesFile mf_demo = new ModulesFile();
			ModulesFile mf_good = new ModulesFile();
			ModulesFile mf_bad = new ModulesFile();

			mf_opt.setModelType(ModelType.DTMC);
			mf_demo.setModelType(mf_opt.getModelType());
			mf_good.setModelType(mf_opt.getModelType());
			mf_bad.setModelType(mf_opt.getModelType());

			ArrayList<String> files = new ArrayList<String>();
			String STATE_SPACE = "//home/zekunzhou/workspace/Safety-AI-MDP/cirl/state_space";
			String OPT_POLICY = "//home/zekunzhou/workspace/Safety-AI-MDP/cirl/optimal_policy";
			String DEMO_POLICY = "//home/zekunzhou/workspace/Safety-AI-MDP/cirl/demo_policy";
			files.add(STATE_SPACE);
			files.add(OPT_POLICY);
			files.add(DEMO_POLICY);
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
						lines.add(line);
					}
					if (file.equals(STATE_SPACE)) {
						ConstantDef(mf_opt.getConstantList(), lines);
						mf_demo.setConstantList(mf_opt.getConstantList());
						mf_good.setConstantList(mf_opt.getConstantList());
						mf_bad.setConstantList(mf_opt.getConstantList());
						lines.clear();
					}
					if (file.equals(OPT_POLICY)) {
						ParsePolicy(lines);
						FormulaDef(mf_opt.getFormulaList(), policy);
						lines.clear();
					}
					if (file.equals(DEMO_POLICY)) {
						ParsePolicy(lines);
						FormulaDef(mf_demo.getFormulaList(), policy);
						lines.clear();
					}
				} catch (IOException e) {
					e.printStackTrace();
				}
			}

			Module m_opt = Module("grid_world", mf_opt.getConstantList(), mf_opt.getFormulaList());
			mf_opt.addModule(m_opt);
			mf_opt.setInitialStates(new ExpressionLiteral(TypeBool.getInstance(), "x = 0 & y = 0"));

			mf_opt.tidyUp();
			
			prism.loadPRISMModel(mf_opt);

			PrintStream ps_console = System.out;
			PrintStream ps_file = new PrintStream(new FileOutputStream(
					new File("//home/zekunzhou/workspace/Safety-AI-MDP/cirl/grid_world.pm")));
			System.setOut(ps_file);
			System.out.println(mf_opt);
			
			System.setOut(ps_console);
			/**
			System.out.println(mf_opt);
			ModulesFile modulesFile = prism
					.parseModelFile(new File("//home/zekunzhou/workspace/Safety-AI-MDP/cirl/grid_world.pm"));
			prism.loadPRISMModel(modulesFile);
			//Parse and load a properties model for the model
			PropertiesFile propertiesFile = prism.parsePropertiesString(mf_opt, "P=? [true U<=64 x = " + Integer.toString(x_l_0) + " & y = " + Integer.toString(y_l_0) + "]");
			Result result = prism.modelCheck(propertiesFile, propertiesFile.getPropertyObject(0));
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