import subprocess
import argparse
import time
import datetime
from tqdm import tqdm
from glob import glob
import os


def terra_lookxml(path, apttern='TDX1_SAR__EEC_SE___SM_D_SRA'):
    ''''Function tht scans directory and returns Terra-SAR-X xml file
        path: the root file for extracted image
        pattern: the pattern to filter files and folders, may vary as per product type
        returns: xml file description fith full path
    '''
    
    xml_container = []
    for root, subdir, files in os.walk(path):
        if apttern in root:
            for file in files:
                if apttern in file:
                    xml_container.append(root + '/' + file)
    assert len(xml_container) == 1, f'found multiple xmls or there is no xml file in {path}'
    
    return xml_container[0]


def TerraSAR_GPT(data_dir, out_dir, graph, gpt_path):
    '''TerraSAR-X processing using ESA SNAP graph processing tool
        data_dir: path contains all extracted files, 
        out_dir: Output directory to save processsed files
        graph: ESA snap processing graph for specific workflows
        gpt_path: ESA snap graph processing tool, mainly from the bin folder of installation directory
    '''
    
    warmup = time.time()
    folds = os.listdir(data_dir)
    for fold in folds:
        im_dir = data_dir + f'/{fold}'
        prod_xml = terra_lookxml(im_dir)   # returns the xml definition of the image
        name = os.path.split(prod_xml)[1][:-4] + '.tif'
        save_path = out_dir + f'/{name}'

        with open(graph, 'r') as grf:
            grdata = grf.read()
            grdata = grdata.replace('INPUT', prod_xml)
            grdata = grdata.replace('OUTPUT', save_path)  # New will be removed its for testing
            root, file = os.path.split(graph)
            graph2run = root + '/graph_cpy.xml'
            with open(graph2run, 'w') as grfn:
                grfn.write(grdata)
        args = [gpt_path, graph2run, '-c', '7G', '-q', str(8)] #
        start = time.time()
        print('\n Started processing graph at {}'.format(time.ctime(start)))
        process = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
        stdout = process.communicate()[0]
        print('\n Output {} :'.format(stdout))
        os.remove(graph2run)
    marathon = (time.time() - warmup) / 60
    print('\n Processing of {} raw files is done within {} minutes'.format(counter, marathon))

def argumentParser():
    parser = argparse.ArgumentParser(description='Graph to batch process terraSARX files')
    parser.add_argument('--data', help='data to process', type=str, default='terrasar', required=False)
    parser.add_argument('--data_dir', help='Data directory', type=str, required=True)
    parser.add_argument('--save_dir', help='Data to save results', type=str, required=True)
    parser.add_argument('--graph', help='processing graph pipline', type=str, required=True)
    parser.add_argument('--gpt_path', help='snap graph processing engine location',
                        type=str, 
                        default='C:/Program Files/snap/bin/gpt.exe')
    my_args = parser.parse_args()
    
    return my_args


if __name__ == '__main__':
    args = argumentParser()
    if args.data == 'terrasar':
        TerraSAR_GPT(data_dir=args.data_dir,
                     out_dir=args.save_dir,
                     graph=args.graph,
                     gpt_path=args.gpt_path)
    else:
        Sentinel_GPT(data_dir=args.data_dir,
                     out_dir=args.save_dir,
                     graph=args.graph,
                     gpt_path=args.gpt_path)