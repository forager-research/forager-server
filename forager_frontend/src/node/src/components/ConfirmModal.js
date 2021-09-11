import React from "react";
import {
  Modal,
  ModalHeader,
  ModalFooter,
  ModalBody,
} from "reactstrap";

const ConfirmModal = ({
  isOpen,
  toggle,
  message,
  confirmBtn,
  cancelBtn,
}) => {
  return (
    <Modal
      isOpen={isOpen}
      modalTransition={{ timeout: 25 }}
      backdropTransition={{ timeout: 75 }}
    >
      <ModalHeader>Confirmation Required</ModalHeader>
      <ModalBody>{message}</ModalBody>
      <ModalFooter>
        {cancelBtn}{" "}
        {confirmBtn}
      </ModalFooter>
    </Modal>
  );
};

export default ConfirmModal;
