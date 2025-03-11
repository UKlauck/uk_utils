import logging
import os
import re

try:
    import pwd
    pwd_availabe = True
except ImportError:
    import getpass
    from pathlib import Path
    pwd_availabe = False

# ------------------------------------------------------------------------------
def split_nb_name(fn):
    re_parts = [re.compile(r'[A-Z]+_'),  # module name
                re.compile(r'\d+[a-z]*[-_]'),  # module number
                re.compile(r'\d+[a-z]*_'),  # file number
                re.compile(r'\w+'),  # file name
                re.compile(r'\.\w+')  # extension
                ]

    re_results = [None, None, None, None, None]

    ok = True

    m_name = None
    m_number = None
    m_filenumber = None
    m_filename = None
    m_ext = None

    for i, regex in enumerate(re_parts):
        res = regex.match(fn)
        if res is None:
            ok = False
            break
        else:
            re_results[i] = res.group(0)
            fn = fn[res.span(0)[1]:]

    return ok, re_results

# ------------------------------------------------------------------------------
def build_nb_name(m_name, m_number, f_number, f_name, f_ext):
    # Check and correct module name. It must end with a _
    if m_name[-1] != '_':
        m_name = m_name + '_'

    # Check and correct module number. It must end with a -
    m_number = check_number(m_number)

    return m_name + m_number + f_number + f_name + f_ext


# ------------------------------------------------------------------------------
def check_number(nr):
    if type(nr) is int:
        nr = f'{nr:2d}-'
    elif type(nr) is str:
        if nr[-1] == '_':
            nr = nr[:-1] + '-'
        elif nr[-1] != '-':
            nr = nr + '-'

    return nr


# ------------------------------------------------------------------------------
def change_module_name(new_name=None, old_name=None, verbose=False):
    if new_name is None:
        return
    if old_name is not None:
        if old_name[-1] != '_':
            old_name = old_name + '_'

    for fn in os.listdir():
        ok, (m_name, m_number, f_number, f_name, f_ext) = split_nb_name(fn)
        if ok:
            change_it = True
            if old_name is not None:
                if m_name != old_name:
                    change_it = False

            if change_it:
                new_fname = build_nb_name(new_name, m_number, f_number, f_name, f_ext)
                os_cmd = f'mv {fn} {new_fname}'
                if verbose:
                    print(os_cmd)
                os.system(os_cmd)


# ------------------------------------------------------------------------------
def change_module_number(new_nr=None, old_nr=None, module=None, verbose=False):
    if new_nr is None:
        return

    new_nr = check_number(new_nr)

    if old_nr is not None:
        old_nr = check_number(old_nr)

    for fn in os.listdir():
        ok, (m_name, m_number, f_number, f_name, f_ext) = split_nb_name(fn)
        if ok:
            change_it = True
            if old_nr is not None:
                if m_number != old_nr:
                    change_it = False
            if module is not None:
                if m_name != module:
                    change_it = False

            if change_it:
                new_fname = build_nb_name(m_name, new_nr, f_number, f_name, f_ext)
                os_cmd = f'mv {fn} {new_fname}'
                if verbose:
                    print(os_cmd)
                os.system(os_cmd)


# ------------------------------------------------------------------------------
def get_username():
    """
    Returns the current user name
    """
    if pwd_availabe:
        return pwd.getpwuid(os.getuid()).pw_name
    else:
        return getpass.getuser()

# ------------------------------------------------------------------------------
def get_userhome():
    """
    Returns the full path name of the users's home directory
    """
    if pwd_availabe:
        return pwd.getpwuid(os.getuid()).pw_dir
    else:
        return str(Path.home())

# ------------------------------------------------------------------------------
def get_course_home():
    """
    Returns the full path name of the base for courses (~home_dir/Vorlesungen/Eigene Unterlagen)
    """
    return os.sep.join([get_userhome(), 'Vorlesungen', 'Eigene Unterlagen'])


# ------------------------------------------------------------------------------
def create_new_course(course_name):
    """
    Creates a new course directory with some typical subdirectories if it does not yet exist.
    Will do nothing for existing course directories

    Parameters
    ----------
    course_name:   str  Name of the course directory
    """
    new_path = os.sep.join([get_course_home(), course_name])

    if os.path.exists(new_path):
        print(f'Course directory {new_path} already exists')
    else:
        os.path.os.mkdir(new_path)

    for subdir in ['Slides', 'Examination', 'Assignments', 'Exercises', 'Examples']:
        new_subdir = os.sep.join([new_path, subdir])
        if os.path.exists(new_subdir):
            print(f'Course subdirectory {new_subdir} already exists')
        else:
            os.path.os.mkdir(new_subdir)


# ------------------------------------------------------------------------------
def copy_chapter(from_course=None, to_course=None,
                 chapter=None, new_chapter=None,
                 old_name=None, new_name=None,
                 institution=None,
                 slide_title=None,
                 overwrite_existing=False,
                 log_level=logging.ERROR):
    """Copies all files from a given chapter to a chapter in a new course

    from_course  (str)      Source
    to_course    (str)      Destination
    chapter      (str, int) Chapter number in string format , '00'-'99'
    new_chapter  (str, int) New chapter number
    old_name     (str)      Old name of the notebooks. If None, name will not be checked
    new_name     (str)      New name of the notebooks
    institution  (str)      Prefix for LaTeX-class (GSO or HSAA)
    slide_title  (str)      title of the slide (\title{} tag)
    overwrite_existing (boolean) If True, existing chapters will be overwritten
    log_level    (int)      loglevel as understood by logging library

    Format of notebook filenames:
      name_nn-nnc_[s]+.ipynb

         nn:   Chapter number, e.g. 00, 01, ...
         nnc:  Number of notebook, e.g. 01a, 02, ...

    """

    """ TODO
    Include institutions UWC PVA 
    """
    logger = logging.getLogger("CU")
    handler = logging.StreamHandler()
    #handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
    handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(log_level)

    curdir = os.path.os.getcwd()

    errors = False

    if from_course is None:
        logger.fatal('You must specifiy a course to copy from')
        errors = True

    if to_course is None:
        logger.fatal('You must specifiy a course to copy to')
        errors = True

    if chapter is None:
        logger.fatal('You must specifiy a chapter to copy')
        errors = True
    elif type(chapter) is int:
        chapter = f'{chapter:02d}'

    if errors:
        return

    if new_chapter is None:
        new_chapter = chapter
    elif type(new_chapter) is int:
        new_chapter = f'{new_chapter:02d}'

    if new_name is None:
        new_name = to_course

    errors = False

    from_home = os.sep.join([get_course_home(), from_course])
    to_home = os.sep.join([get_course_home(), to_course])

    if not os.path.exists(from_home):
        logger.fatal(f'Source course {from_course} does not exist')
        errors = True

    if not os.path.exists(to_home):
        logger.fatal(f'Target course {to_course} does not exist')
        errors = True

    if errors:
        logger.fatal(f'Giving up ')
        return

    # Copy all chapter-based directories
    for subdir in ['Exercises', 'Examples']:
        source = os.sep.join([from_home, subdir, 'chap'+chapter])
        target = os.sep.join([to_home, subdir, 'chap'+new_chapter])

        logger.info(f'Copying from {source} to {target}')

        if not os.path.exists(source):
            logger.info(f'Source chapter {source} does not exist')
            continue

        if os.path.exists(target):
            if overwrite_existing:
                # Clear the directory
                shell_cmd = f'rm -rf "{target}"'
                logger.info(f'Deleting directory {target}')
                os.path.os.system(shell_cmd)
            else:
                logger.warning(f'Target chapter {target} already exists. No copy performed')
                continue

        # Copy the chapter ...
        os.path.os.mkdir(target)
        for dir_entry in os.scandir(source):
            if dir_entry.is_file():
                f = dir_entry.name
                shell_cmd = f'cp "{os.sep.join([source, f])}" "{target}"'
                logger.debug(shell_cmd)
                os.path.os.system(shell_cmd)
                #comm = subprocess.Popen([shell_cmd], stdout=subprocess.PIPE, shell=True)
                #data = comm.communicate()
                # print(data)
            else:
                if dir_entry.name.find('.ipynb_checkpoints') < 0:    # No warnings for .ipynb_checkpoints directories
                    logger.warning(f'{dir_entry.name} is a directory. No copy performed')

        # ... and rename the modules
        os.path.os.chdir(target)
        change_module_name(new_name=new_name, old_name=old_name)
        change_module_number(new_nr=new_chapter)

    # Copy slides
    subdir = 'Slides'
    source = os.sep.join([from_home, subdir])
    target = os.sep.join([to_home, subdir])
    logger.debug(f'Copying slides from {source} to {target}')

    # Change slides style according to the institution settings
    if institution == 'GSO' or institution == 'HSAA':
        # Copy correct course_specific.tex version
        source_file = f'{source}/course_specific_{institution}.tex'
        target_file = f'{target}/course_specific.tex'
        if not os.path.exists(source_file):
            logger.warning(f'Source file {source_file} does not exist.')
        else:
            shell_cmd = f'cp "{source_file}" "{target_file}"'
            logger.debug(shell_cmd)
            os.path.os.system(shell_cmd)
    elif institution is not None:
        logger.error(f'Institution {institution} not defined')

    # Copy the slide files
    for dir_entry in os.scandir(source):
        # forget about all non-tex files in the slides directory
        if dir_entry.is_file():
            f = dir_entry.name
            if not f.endswith('.tex'):
                continue

            source_file = os.sep.join([source, f])
            file_copied = False

            if f.endswith(f'{chapter}_Beamer.tex'):
                # Copy the file
                target_file = os.sep.join([target, f'{new_name}_{new_chapter}_Beamer.tex'])
                shell_cmd = f'cp "{source_file}" "{target_file}"'
                logger.info(f)
                os.path.os.system(shell_cmd)
                file_copied = True

                # \input correct header
                search_string = r'input{.*Header_B}'
                repl_string = r'input{' + institution + r'_Header_B}'
                shell_cmd = 'sed -i.bak "1,2s/' + search_string + '/' + repl_string + '/" ' + f'"{target_file}"'
                logger.debug(shell_cmd)
                os.path.os.system(shell_cmd)
            elif f.endswith('{chapter}_Handout.tex'):
                # Copy the file
                target_file = os.sep.join([target, f'{new_name}_{new_chapter}_Handout.tex'])
                shell_cmd = f'cp "{source_file}" "{target_file}"'
                logger.debug(shell_cmd)
                os.path.os.system(shell_cmd)
                file_copied = True

                # \input correct header
                search_string = r'input{.*Header_H}'
                repl_string = r'input{' + institution + r'_Header_H}'
                shell_cmd = 'sed -i.bak "1,2s/' + search_string + '/' + repl_string + '/" ' + f'"{target_file}"'
                logger.debug(shell_cmd)
                os.path.os.system(shell_cmd)
            else:
                # Don't copy regular files
                # Simple tex-file, just copy it
                target_file = os.sep.join([target, f])
                #shell_cmd = f'cp "{source_file}" "{target_file}"'
                #logger.info(f)
                #os.path.os.system(shell_cmd)

            if file_copied and (target_file.endswith('_Beamer.tex') or target_file.endswith('_Handout.tex')):
                search_string = r"\\title{.*}"
                repl_string = r"\\title{" + slide_title + r"}"
                shell_cmd = "sed -i.bak '1,5s/" + search_string + "/" + repl_string + "/' " + f"'{target_file}'"
                logger.debug(shell_cmd)
                os.path.os.system(shell_cmd)

                search_string = r"\\setbeamercolor{background canvas}{bg.*}"
                if institution == 'GSO':  # set background image for title slide
                    repl_string = r"\\setbeamercolor{background canvas}{bg=GSObackgroundTitle}"
                    shell_cmd = "sed -i.bak '1,13s/" + search_string + "/" + repl_string + "/' " + f"'{target_file}'"
                    logger.debug(shell_cmd)
                    os.path.os.system(shell_cmd)
                else: # institution == 'HSAA':  # reset background image
                    repl_string = r"\\setbeamercolor{background canvas}{bg=}"
                    shell_cmd = "sed -i.bak '1,12s/" + search_string + "/" + repl_string + "/' " + f"'{target_file}'"
                    logger.debug(shell_cmd)
                    os.path.os.system(shell_cmd)

                if slide_title is not None:
                    # Change the \title tag
                    for f in os.listdir(target):
                        if f.endswith('{chapter}_Beamer.tex') or f.endswith('{chapter}_Handout.tex'):
                            search_string = r"\\title{.*}"
                            repl_string = r"\\title{"+slide_title+r"}"
                            shell_cmd = "sed -i.bak '1,5s/"+search_string+"/"+repl_string+"/' " +f"'{target_file}'"
                            logger.debug(shell_cmd)
                            os.path.os.system(shell_cmd)
            # Delete eventually created backup file
            try:
                if os.path.exists(f'{target_file}.bak'):
                    shell_cmd = f"rm '{target_file}'.bak"
                    os.path.os.system(shell_cmd)
                    logger.debug(shell_cmd)
            except:
                pass

            # Copy the file if it is particular for the institution's course
            if not file_copied and institution is not None:
                if institution in f:
                    target_file = os.sep.join([target, f])
                    shell_cmd = f'cp "{source_file}" "{target_file}"'
                    logger.info(f)
                    os.path.os.system(shell_cmd)

        else:
            if dir_entry.name.find('.ipynb_checkpoints') < 0:    # No warnings for .ipynb_checkpoints directories
                logger.warning(f'{dir_entry.name} is a directory. No copy performed')


    # Copy all non chapter-based directories
    for subdir in ['Assignments']:
        source = os.sep.join([from_home, subdir])
        target = os.sep.join([to_home, subdir])

        try:
            for dir_entry in os.scandir(source):
                if dir_entry.is_file():
                    f = dir_entry.name
                    shell_cmd = f'cp "{os.sep.join([source, f])}" "{target}"'
                    logger.debug(shell_cmd)
                    os.path.os.system(shell_cmd)
                else:
                    if dir_entry.name.find('.ipynb_checkpoints') < 0:    # No warnings for .ipynb_checkpoints directories
                        logger.warning(f'{dir_entry.name} is a directory. No copy performed')
        except FileNotFoundError:
            logger.warning(f'No files copied from dir {subdir}')

    os.path.os.chdir(curdir)


# ------------------------------------------------------------------------------
class CourseGenerator:

    def __init__(self, content_dict=None):

        self.p_dict = {
            'work_dir': get_userhome() + f'/Vorlesungen/Eigene Unterlagen',
            'source_course': 'ML',
            'target_course': None,
            'prefix': None,
            'defines': None,
            'header_B': 'Header_B',
            'header_H': 'Header_H',
            'language': 'english',
            'title': None,
            'subtitle': None,
            'title_page': False,
            'loglevel': logging.ERROR,
        }

        if content_dict is not None:
            self.contents = content_dict
        else:

            self.contents = {
                'Outline': ('00', ('ML_Outline',)),
                'OutlineBoschLT': ('00', ('ML_Outline_Bosch_Longterm',)),
                'OutlineBoschDS': ('00', ('ML_Outline_Bosch_DS',)),
                'OutlineGCAI': ('00', ('ML_Outline_GCAI',)),
                'OutlineGCML': ('00', ('ML_Outline_GCML',)),
                'OutlineHSAA': ('00', ('ML_Outline_HSAA',)),

                'Intro': ('01', ('ML_Intro',)),
                'IntroBoschLT': ('01', ('ML_Intro_Bosch_Longterm',)),
                'Workflow': (None, ('ML_Workflow',)),

                'LinearMethods': ('02', ('ML_LinReg', 'ML_LogReg')),
                'LinearMethodsRecap': ('02', ('ML_LinRegRecap', 'ML_LogReg')),

                'LinearRegression': ('02a', ('ML_LinReg',)),
                'LinReg': ('02a', ('ML_LinReg',)),
                'LinRegRecap': ('02a', ('ML_LinRegRecap',)),
                'LogisticRegression': ('02b', ('ML_LogReg',)),
                'LogReg': ('02b', ('ML_LogReg',)),

                'kNN': ('03', ('ML_kNN',)),
                'DTree': ('04', ('ML_DTree',)),
                'Evaluation': ('05', ('ML_Evaluation', )),
                'FeatureSelection': ('06', ('ML_FeatureSelection', )),
                'DimReduction': ('07', ('ML_FeatureExtraction', 'ML_FeatureVisualization')),

                'BayesDA': ('09', ('ML_Bayes', )),

                'SVM': ('11', ('ML_SVM',)),
                'Cluster': ('12', ('ML_Cluster',)),
                'TimeSeries': ('13', ('DA_TimeSeries1', 'DA_Fourier', 'DA_TSDecomposition',
                                      'DA_TSAR', 'DA_TSMA', 'DA_TSARMA', 'DA_TSARIMA')),
                'TimeSeries2': ('13a', ('DA_TimeSeries2', 'DA_Fourier', 'DA_TSDecomposition2',
                                      'DA_TSARIMA2', 'DA_TSForecasting', 'DA_TSSummary',
                                        'DA_TSLibs')),
                'NN': ('15', ('ML_NN_Perceptron', 'ML_NN_Activation', 'ML_NN_Loss',
                              'ML_NN_PerceptronRegression', 'ML_NN_PerceptronClassification',
                              'ML_NN_Learning', 'ML_MLP', 'ML_MLP_Learning')),
                'NNOptimization': ('15a', ('ML_NN_Optimization',)),

                'DLIntro': ('21', ('ML_DLIntro',)),
                'CNNBasics': ('22', ('ML_CNN_BasicElements', 'ML_CNN_Basics_Conv', 'ML_CNN_Basics_Pooling', 'ML_CNN_Basics_Example')),
                'CNNOptimization': ('23', ('ML_SourcesOfError',)),
                'CNNClassicalArchitectures': ('24', ('ML_ClassicalCNN',)),
                'TransferLearning': ('25', ('ML_TransferLearning',)),
                'AE': ('26', ('ML_Autoencoders',)),
                'Autoencoders': ('26', ('ML_Autoencoders',)),
                'DLTimeSeries': ('27', ('ML_TimeSeriesDL',)),

                # Data Analytics course
                'DA_Outline': ('50', ('DA_Outline',)),
                'DA_Intro': ('01', ('DA_Intro', 'DA_Data', 'DA_Visualization')),
                'DA_Basics': ('51', ('DA_BasicStatistics', )),
                'DA_Cleaning': ('52', ('DA_DataCleaning',)),
                'DA_Hypotheses': ('53', ('DA_Hypotheses',)),
            }

        self.logger = logging.getLogger("CU")
        handler = logging.StreamHandler()
        # handler.setFormatter(logging.Formatter("%(asctime)s|%(levelname)s|%(name)s|%(message)s"))
        handler.setFormatter(logging.Formatter("%(levelname)s | %(name)s | %(message)s"))
        self.logger.addHandler(handler)


    def createChapter(self, c_nr=1, c_content=None, c_title=None):
        self.logger.setLevel(self.getParameter('loglevel'))
        if c_title is not None:
            self.setParameters({'subtitle': c_title})

        self.createAllOutputDirectories()
        try:
            source_chapter, tex_files = self.contents[c_content]
        except KeyError:
            source_chapter = None
            tex_files = None
            pass

        target_chapter = f'{c_nr:02d}'

        if tex_files is not None:
            self.generateLaTeX(target_chapter, tex_files)

        if source_chapter is not None:
            self.copyNotebooks('Exercises', source_chapter, target_chapter)
            self.copyNotebooks('Examples', source_chapter, target_chapter)


    def generateLaTeX(self, chapter, tex_files):
        header_B = self.getParameter('header_B')
        title = self.getParameter('title', ' ')
        subtitle = self.getParameter('subtitle', ' ')
        language = self.getParameter('language', 'english')
        defines = self.getParameter('defines', None)

        filename = f"{self.outputPath()}/Slides/{self.getParameter('prefix')}_{chapter}_Beamer.tex"
        self.logger.debug(f'Creating {filename}')

        with open(filename, 'wt', ) as outfile:
            # Import header
            outfile.write(r'\input{' + header_B + '}\n\n')

            # course specific
            # outfile.write(r'\include{course_specific}' + '\n\n')

            outfile.write(r'''\graphicspath{{../../Abbildungen/}{../../}}
\makeatletter
\providecommand*{\input@path}{}
\edef\input@path{{../../ML/Slides/}\input@path}% prepend
\makeatother''')
            outfile.write('\n\n')

            # Optional definition
            if defines is not None:
                for d in defines:
                    outfile.write(r'\edef' + f'\\{d}' +'{}')
                    outfile.write('\n')
                outfile.write('\n')

            # title and subtitle
            outfile.write(r'\title{' + title + '}\n')
            outfile.write(r'\subtitle{' + subtitle + '}\n\n')

            # begin document
            outfile.write(r'\begin{document}' + '\n\n')

            # language
            outfile.write(r'\selectlanguage{' + language + '}\n\n')

            # beamer color and title page and frame
            tp = self.getParameter('title_page', False)
            if tp:
                bg = self.getParameter('title_background')
                if bg is not None:
                    outfile.write(r'\setbeamercolor{background canvas}{bg=' + bg + '}\n')

                outfile.write(r'\begin{frame}' + '\n   ' + r'\titlepage' + '\n' + r'\end{frame}' + '\n')

                if bg is not None:
                    outfile.write(r'\setbeamercolor{background canvas}{bg=}' + '\n')

            outfile.write('\n')

            # section
            outfile.write(r'\section{' + subtitle + '}\n\n')

            # input tex files
            for f in tex_files:
                outfile.write(r'\input{' + f + '.tex}\n')
            outfile.write('\n')

            # end document
            outfile.write(r'\end{document}' + '\n')

    def getParameter(self, p_name, p_default=None):
        try:
            p_value = self.p_dict[p_name]
        except KeyError:
            p_value = p_default
        return p_value

    def setParameters(self, p_dict):
        for (k, v) in p_dict.items():
            self.p_dict[k] = v

    def inputPath(self):
        return f"{self.getParameter('work_dir')}{os.sep}{self.getParameter('source_course')}"

    def outputPath(self):
        return f"{self.getParameter('work_dir')}{os.sep}{self.getParameter('target_course')}"

    def createAllOutputDirectories(self):
        outpath = self.outputPath()
        for d in ('Slides', 'Examples', 'Exercises', 'Assignments'):
            os.path.os.makedirs(f'{outpath}{os.sep}{d}', exist_ok=True)

    def copyNotebooks(self, directory, source_chapter, target_chapter):
        inPath = f"{self.inputPath()}{os.sep}{directory}{os.sep}chap{source_chapter}"
        outPath = f"{self.outputPath()}{os.sep}{directory}{os.sep}chap{target_chapter}"

        if os.path.exists(inPath):
            os.path.os.makedirs(f'{outPath}', exist_ok=True)
            for dir_entry in os.scandir(inPath):
                if dir_entry.is_file():
                    f = dir_entry.name
                    shell_cmd = f'cp "{os.sep.join([inPath, f])}" "{outPath}{os.sep}."'
                    self.logger.debug(shell_cmd)
                    os.path.os.system(shell_cmd)
                else:
                    if dir_entry.name.find('.ipynb_checkpoints') < 0:    # No warnings for .ipynb_checkpoints directories
                        self.logger.warning(f'{dir_entry.name} is a directory. No copy performed')

        else:
            self.logger.warning(f'No notebooks available in {inPath}')

        # Now change notebook names to match new prefix and module number
        if os.path.exists(outPath):
            oldPath = os.path.os.getcwd()
            os.path.os.chdir(outPath)
            for dir_entry in os.scandir(outPath):
                if dir_entry.is_file():
                    fn = dir_entry.name

                    ok, (m_name, m_number, f_number, f_name, f_ext) = split_nb_name(fn)
                    if ok:
                        new_name = self.getParameter('prefix')
                        new_nr = target_chapter
                        new_fname = build_nb_name(new_name, new_nr, f_number, f_name, f_ext)
                        os_cmd = f'mv {fn} {new_fname}'
                        self.logger.debug(os_cmd)
                        os.system(os_cmd)
            os.path.os.chdir(oldPath)







