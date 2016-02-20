import numpy as np
class ISHNE(object):
    def __init__(self, filename):
        self.filename = filename
        self.leads = None, None, None

    def read(self):
        with open(self.filename, 'rb') as f:
            # magic number
            magicnumber = np.fromfile(f, dtype = np.dtype('a8'), count = 1)[0]
        
            # check sum
            chesksum = np.fromfile(f, dtype = np.uint16, count = 1)[0]
    
            #header
            Var_length_block_size = np.fromfile(f, dtype = np.int32, count = 1)[0] 
            Sample_Size_ECG =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
            Offset_var_length_block =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
            Offset_ECG_block =  np.fromfile(f, dtype = np.int32, count = 1)[0] 
            File_version =  np.fromfile(f, dtype = np.int16, count = 1)[0] 
            First_name =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
            Last_name =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
            ID = np.fromfile(f, dtype = np.dtype('a20'), count = 1)[0]
            Sex = np.fromfile(f, dtype = np.int16, count = 1)[0] 
            Race = np.fromfile(f, dtype = np.int16, count = 1)[0] 
            Birth_Date = np.fromfile(f, dtype = np.int16, count = 3) 
            Record_Date =  np.fromfile(f, dtype = np.int16, count = 3) 
            File_Date =  np.fromfile(f, dtype = np.int16, count = 3) 
            Start_Time =  np.fromfile(f, dtype = np.int16, count = 3) 
            nbLeads = np.fromfile(f, dtype = np.int16, count = 1)[0] 
            Lead_Spec = np.fromfile(f, dtype = np.int16, count = 12) 
            Lead_Qual = np.fromfile(f, dtype = np.int16, count = 12) 
            Resolution = np.fromfile(f, dtype = np.int16, count = 12) 
            Pacemaker = np.fromfile(f, dtype = np.int16, count = 1)[0] 
            Recorder =  np.fromfile(f, dtype = np.dtype('a40'), count = 1)[0]
            Sampling_Rate = np.fromfile(f, dtype = np.int16, count = 1)[0] 
            Propreitary  = np.fromfile(f, dtype = np.dtype('a80'), count = 1)[0]
            Copyright =  np.fromfile(f, dtype = np.dtype('a80'), count = 1)[0]
            Reserved =  np.fromfile(f, dtype = np.dtype('a88'), count = 1)[0]
    
            # read Variable length block
            if (Var_length_block_size >0):
                dt = dtype((str,Var_length_block_size))
                varblock = np.fromfile(f, dtype = dt, count = 1)[0]
        
            # ECG data
            Sample_per_lead = Sample_Size_ECG/nbLeads
        
            ecgSig = np.zeros((nbLeads, Sample_per_lead))
        
            # 1000 is temp, it is actually Sample_per_lead
            # for i in range(1000):
            #     for j in range(nbLeads):
            #         ecgSig[j][i] =  np.fromfile(f, dtype = np.int16, count = 1)[0]
            ecg_dtype = np.dtype([('samples', np.int16, nbLeads)])
            ecgSig = np.fromfile(f, dtype=ecg_dtype, count=int(Sample_per_lead))
            # print 'Sample Size = %d' % Sample_Size_ECG
            # print 'Number of Leads = %d' % nbLeads
            # print 'Resolution = %s' % Resolution
            # print 'Sampling Rate = %d' % Sampling_Rate
            
            y1 = ecgSig['samples'][:, 0] * (Resolution[0]/1000000.0)
            y2 = ecgSig['samples'][:, 1] * (Resolution[1]/1000000.0)
            y3 = ecgSig['samples'][:, 2] * (Resolution[2]/1000000.0)
            self.leads = y1, y2, y3
            self.sampling_rate = Sampling_Rate
