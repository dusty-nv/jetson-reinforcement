/*
 * deepRL
 */

#ifdef USE_PYTHON

#include <stdio.h>

#include <Python.h>
#include <TH/TH.h>
#include <THC/THC.h>


int main( int argc, char** argv )
{
	PyObject *pName, *pModule, *pDict, *pFunc;
	PyObject *pArgs, *pValue;

	printf("deepRL-console (python)\n");

	if (argc < 3) {
		fprintf(stderr,"Usage: call pythonfile funcname [args]\n");
		return 1;
	}

	Py_Initialize();

	// setup arguments to python script
	int py_argc = 3;
	char* py_argv[3];

	py_argv[0] = argv[1];
	py_argv[1] = "--foo";
	py_argv[2] = "--bar";

	PySys_SetArgv(py_argc, py_argv);
	//PySys_SetArgv(argc, argv);

	pName = PyString_FromString(argv[1]);

	// load the script
	pModule = PyImport_Import(pName);
	Py_DECREF(pName);

	if (pModule != NULL) 
	{
		pFunc = PyObject_GetAttrString(pModule, argv[2]);
		/* pFunc is a new reference */

		if (pFunc && PyCallable_Check(pFunc)) 
		{
			pArgs = PyTuple_New(argc - 3);
			
			for (int i = 0; i < argc - 3; ++i) 
			{
				pValue = PyInt_FromLong(atoi(argv[i + 3]));

				if (!pValue) 
				{
					Py_DECREF(pArgs);
					Py_DECREF(pModule);
					fprintf(stderr, "Cannot convert argument\n");
					return 1;
				}

				/* pValue reference stolen here: */
				PyTuple_SetItem(pArgs, i, pValue);
            	}

			// call the function
			pValue = PyObject_CallObject(pFunc, pArgs);

			Py_DECREF(pArgs);

			if (pValue != NULL) 
			{
				printf("Result of call: %ld\n", PyInt_AsLong(pValue));
				Py_DECREF(pValue);
			}
			else 
			{
				Py_DECREF(pFunc);
				Py_DECREF(pModule);
				PyErr_Print();
				fprintf(stderr,"Call failed\n");
				return 1;
			}
		}
		else 
		{
			if (PyErr_Occurred())
				PyErr_Print();

			fprintf(stderr, "Cannot find function \"%s\"\n", argv[2]);
		}

		Py_XDECREF(pFunc);
		Py_DECREF(pModule);
	}
	else 
	{
		PyErr_Print();
		fprintf(stderr, "Failed to load \"%s\"\n", argv[1]);
		return 1;
	}

	Py_Finalize();
	return 0;
}

#endif

