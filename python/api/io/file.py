# Built-in imports

from api import subp
from api import platform
# from api import logging
from api import overload
from api import Union
# from typing import overload, Union, Sequence

# Third-party imports
from api import qtc




class FileIO():
	"""Class for cross platform file managment based on QtCore.

	Convinience methods for automating file - related common tasks like listing files in directories,
	creating new directories, copying files, etc.

	This class ensures path '/' foward slash compability on windows.

	"""

	home = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.HomeLocation)
	desktop = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DesktopLocation)
	download = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DownloadLocation)
	documents = qtc.QStandardPaths.writableLocation(qtc.QStandardPaths.DocumentsLocation)

	osname = platform.system()
	# log = logging.getLogger("FileIO")


	@classmethod
	def __str__(cls) -> str:
		return f"FileIO - cross platform file operation helper."


	@overload
	@classmethod
	def exists(cls, path:qtc.QFileInfo) -> bool: ...
	@overload
	@classmethod
	def exists(cls, path:qtc.QDir) -> bool: ...
	@classmethod
	def exists(cls, path:str) -> bool:
		"""Validate the specified file.

		Args:
			path (str): Path to the object.

		Returns:
			fileInfo (QFileInfo): If the specified object exists it will be returned,
				if it does not exist False will be returned instead.

		"""
		if isinstance(path, str): path = qtc.QFileInfo(path)
		if path.exists():	return True
		return False


	@overload
	@classmethod
	def mkdir(cls, path:qtc.QFileInfo) -> bool: ...
	@overload
	@classmethod
	def mkdir(cls, path:qtc.QDir) -> bool: ...
	@classmethod
	def mkdir(cls, path:str) -> bool:
		"""Creates the specified directory if it does not already exist.

		Args:
			path (string): The path for the directory to be created.

		Returns:
			bool: True if the operation was successful, False if an	error occured during the operation.

		"""
		if isinstance(path, qtc.QFileInfo):
			path = qtc.QDir(path.absoluteDir())
		elif isinstance(path, str):
			path = qtc.QDir(path)
		if not path.exists(): path.mkpath(path.absolutePath())
		return True


	@classmethod
	def listdir(cls,
		path:str,
		nameFilters:list[str]=["*.jpg", "*.png"],
		filters=qtc.QDir.Files,
		includeSubDirectories=qtc.QDirIterator.Subdirectories,
	) -> list[qtc.QFileInfo]:
		"""Returns a list with files contained in the specified directory.

		Args:
			path (string): Path to the directory.
			nameFilters (list): A list with name filters e.g. ['sara*'], ['*.fbx'].
			filters (QDir.Flag): NoFilter, Files, Dirs.
			includeSubDirectories (QDirIterator.IteratorFlag): Whether or not search in	sub-directories,
				Subdirectories - true, NoIteratorFlags - false.

		Returns:
			list_fi (list[QFileInfo]): List with QFileInfo objects that the	directory
				contains,	if the list is empty or the directory does not exis it will	return False.

		"""
		fileInfo = qtc.QFileInfo(path)
		list_fi = []
		if fileInfo.isDir():
			iter = qtc.QDirIterator(fileInfo.absoluteFilePath(), nameFilters, filters, includeSubDirectories)
			while iter.hasNext():
				iter.next()
				fi = iter.fileInfo()
				if fi.fileName() == "." or fi.fileName() == "..": continue ## don't include iterator up / back 
				list_fi.append(fi)
	
		return list_fi


	@classmethod
	def copy(cls, source, destination, overwrite:bool=True) -> bool:
		"""Copies the source file / directory to the destination.

		Args:
			source (str): Source file or directory path.
			destination (str): Destination directory path.
			overwrite (bool): If destination file / directory exists, it will be overwritten.

		Returns:
			bool: True if the operation was successful, False if an	error occured during the
				operation.

		"""
		fileInfo = cls.exists(source)
		if not fileInfo: return False

		# Input can be a directory or a file
		finished = False
		if fileInfo.isDir():
			fileObjs = cls.listdir(source)
			if fileObjs.__len__() != 0:
				for fileObj in fileObjs:
					destinationFilePath = fileObj.filePath().replace(source, destination)
					destinationFile = qtc.QFile(destinationFilePath)
					if overwrite and destinationFile.exists(): destinationFile.remove()
					destinationDir = qtc.QFileInfo(destinationFilePath).dir()
					cls.mkdir(destinationDir)
					finished = qtc.QFile(fileObj.filePath()).copy(destinationFilePath)
				cls.log.info(f'{fileObjs.__len__()} files copied successfully.')
			else:
				cls.log.info('Did not find any files to copy in the given directory')

		if fileInfo.isFile():
			cls.mkdir(destination)
			destinationFilePath = f'{destination}/{fileInfo.fileName()}'
			destinationFile = qtc.QFile(destinationFilePath)
			if overwrite and destinationFile.exists(): destinationFile.remove()
			finished = qtc.QFile(fileInfo.filePath()).copy(destinationFilePath)

		if finished:
			cls.log.info("Copy operation finished")
			return True

		return False


	@overload
	@classmethod
	def remove(cls, path:qtc.QFileInfo) -> bool: ...
	@overload
	@classmethod
	def remove(cls, path:list) -> bool: ...
	@classmethod
	# def remove(cls, path:str) -> bool:
	def remove(cls, path:Union[str, qtc.QFileInfo]) -> bool:
		"""Removes the specified path."""
		if isinstance(path, list): [cls.remove_path(qtc.QFileInfo(indx)) for indx in path]
		elif isinstance(path, str):	path = qtc.QFileInfo(path)
		cls.remove_path(path)
		return True

	@classmethod
	def remove_path(cls, path:qtc.QFileInfo):
		if path.isFile():	qtc.QFile(path.absoluteFilePath()).remove()
		if path.isDir(): path.absoluteDir().removeRecursively()


	@classmethod
	def open(cls, path:str):
		if cls.osname == "Windows":
			subp.Popen(f'explorer "{qtc.QDir.toNativeSeparators(path)}"')
		elif cls.osname == "Darwin":
			subp.Popen(f"open {path}", shell=True)
		elif cls.osname == "Linux":
			subp.Popen(f"gnome-open {path}", shell=True)
		else:
			cls.log.critical(f"Unsupported operating system: '{cls.osname}'.")


	@classmethod
	def unzip(cls, path, outpath:str="./"):
		"""Unarchive the specified file."""
		try:
			subp.run(["tar", "xf", path, "-C", outpath], check=True)
			print(f"Extracted {path} to {outpath}")
		except subp.CalledProcessError as err:
			print(f"Error extracting {path}: {str(err)}")
